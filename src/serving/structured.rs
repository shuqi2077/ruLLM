use super::*;
use crate::GenerationError;
use crate::GenerationStep;
#[cfg(feature = "structured-output")]
mod llg;

#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchema },
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonSchema {
    pub name: String,
    pub description: Option<String>,
    pub schema: Value,
    pub strict: Option<bool>,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Grammar {
    Regex(String),
    Lark(String),
    Llguidance(Value),
}

pub fn requested(request: &GenerateRequest) -> bool {
    request.grammar.is_some()
        || request
            .response_format
            .as_ref()
            .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

pub(super) fn validate(request: &GenerateRequest) -> Result<(), String> {
    if !requested(request) {
        return Ok(());
    }
    #[cfg(not(feature = "structured-output"))]
    return Err("constrained decoding requires the structured-output feature".into());
    #[cfg(feature = "structured-output")]
    {
        if request.grammar.is_some() && request.response_format.is_some() {
            return Err("supply grammar or response_format, not both".into());
        }
        if request.tools_enabled() {
            return Err("response constraints cannot currently be combined with tool calls".into());
        }
        if let Some(ResponseFormat::JsonSchema { json_schema }) = &request.response_format {
            if json_schema.name.is_empty()
                || (!json_schema.schema.is_object() && !json_schema.schema.is_boolean())
            {
                return Err(
                    "json_schema requires a nonempty name and an object or boolean schema".into(),
                );
            }
        }
        Ok(())
    }
}

pub(super) struct Decoder {
    #[cfg(feature = "structured-output")]
    matcher: llguidance::Matcher,
    #[cfg(feature = "structured-output")]
    eos: Vec<i32>,
}

impl Decoder {
    pub(super) fn new(
        request: &GenerateRequest,
        tokenizer: &tokenizers::Tokenizer,
        eos: &[i32],
    ) -> Result<Option<Self>, String> {
        validate(request)?;
        if !requested(request) {
            return Ok(None);
        }
        #[cfg(not(feature = "structured-output"))]
        {
            let _ = (tokenizer, eos);
            unreachable!("validated feature requirement");
        }
        #[cfg(feature = "structured-output")]
        {
            use llguidance::api::TopLevelGrammar;
            let grammar = match (&request.grammar, &request.response_format) {
                (Some(Grammar::Regex(regex)), _) => TopLevelGrammar::from_regex(regex),
                (Some(Grammar::Lark(lark)), _) => TopLevelGrammar::from_lark(lark.clone()),
                (Some(Grammar::Llguidance(grammar)), _) => {
                    serde_json::from_value(grammar.clone()).map_err(|e| e.to_string())?
                }
                (_, Some(ResponseFormat::JsonObject)) => {
                    TopLevelGrammar::from_json_schema(json!({"type":"object"}))
                }
                (_, Some(ResponseFormat::JsonSchema { json_schema })) => {
                    TopLevelGrammar::from_json_schema(json_schema.schema.clone())
                }
                _ => return Err("missing response constraint".into()),
            };
            let eos_id = eos
                .first()
                .copied()
                .ok_or("constrained decoding requires a checkpoint EOS token")?;
            let eos_id = u32::try_from(eos_id).map_err(|e| e.to_string())?;
            if eos_id as usize >= tokenizer.get_vocab_size(true) {
                return Err("checkpoint EOS is outside the tokenizer vocabulary".into());
            }
            let factory =
                llg::build_llg_factory(tokenizer.clone(), eos_id).map_err(|e| e.to_string())?;
            let matcher =
                llg::constraint_from_llg_grammar(&factory, grammar).map_err(|e| e.to_string())?;
            Ok(Some(Self {
                matcher,
                eos: eos.to_vec(),
            }))
        }
    }

    #[cfg(feature = "structured-output")]
    fn bias(&mut self, vocabulary: usize) -> Result<Vec<f32>, GenerationError> {
        let mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|e| GenerationError(e.to_string()))?;
        let accepting = self
            .matcher
            .is_accepting()
            .map_err(|e| GenerationError(e.to_string()))?;
        let mut bias = vec![f32::NEG_INFINITY; vocabulary];
        for (index, value) in bias.iter_mut().enumerate() {
            let allowed = if self.eos.contains(&(index as i32)) {
                accepting
            } else {
                index < mask.len() && mask.is_allowed(index as u32)
            };
            if !allowed {
                continue;
            }
            *value = 0.0;
        }
        if bias.iter().all(|value| *value == f32::NEG_INFINITY) {
            return Err(GenerationError(
                "constraint masks the entire model vocabulary".into(),
            ));
        }
        Ok(bias)
    }

    #[cfg(feature = "structured-output")]
    fn mask<B: Backend>(
        &mut self,
        logits: ruda_tensor::api::Tensor<B, 3>,
    ) -> Result<ruda_tensor::api::Tensor<B, 3>, GenerationError> {
        let [_, _, vocabulary] = logits.dims();
        let bias = self.bias(vocabulary)?;
        let device = logits.device();
        let bias = ruda_tensor::api::Tensor::<B, 1>::from_data(
            ruda_tensor::api::TensorData::new(bias, [vocabulary]),
            &device,
        )
        .reshape([1, 1, vocabulary]);
        Ok(logits + bias)
    }

    #[cfg(feature = "structured-output")]
    pub(super) fn consume(&mut self, token: i32) -> Result<(), GenerationError> {
        let ends_turn = self.eos.contains(&token)
            && self
                .matcher
                .is_accepting()
                .map_err(|e| GenerationError(e.to_string()))?;
        if !self.matcher.is_stopped() && !ends_turn {
            self.matcher
                .consume_token(token as u32)
                .map_err(|e| GenerationError(e.to_string()))?;
        }
        Ok(())
    }

    pub(super) fn mask_values(&mut self, values: &mut [f32]) -> Result<(), GenerationError> {
        #[cfg(feature = "structured-output")]
        {
            let bias = self.bias(values.len())?;
            for (value, bias) in values.iter_mut().zip(bias) {
                if bias == f32::NEG_INFINITY {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
        #[cfg(not(feature = "structured-output"))]
        let _ = values;
        Ok(())
    }

    #[cfg(not(feature = "structured-output"))]
    pub(super) fn consume(&mut self, _token: i32) -> Result<(), GenerationError> {
        Ok(())
    }
}

pub(super) fn step<B: Backend, M: CausalModel<B>>(
    session: &mut CausalGenerationSession<'_, B, M>,
    decoder: &mut Option<Decoder>,
) -> Result<Option<GenerationStep>, GenerationError> {
    #[cfg(feature = "structured-output")]
    if let Some(decoder) = decoder {
        let step = session.step_with_logits_processor(|logits| decoder.mask(logits))?;
        if let Some(step) = &step {
            decoder.consume(step.token_id)?;
        }
        return Ok(step);
    }
    #[cfg(not(feature = "structured-output"))]
    let _ = decoder;
    session.step()
}
