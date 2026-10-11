// Adapted from mistral.rs mistralrs-core/src/pipeline/llg.rs at
// 3f2515e9b5adc2ac44949c128c50a13b15721294.
// Copyright (c) 2024 Eric Buehler. MIT license: LICENSE.mistralrs.
use std::sync::Arc;

use anyhow::Result;
use llguidance::{ParserFactory, api::TopLevelGrammar};
use tokenizers::{Tokenizer, decoders::DecoderWrapper};

pub fn build_llg_factory(mut tokenizer: Tokenizer, eos: u32) -> Result<Arc<ParserFactory>> {
    let decoder = match tokenizer.get_decoder() {
        Some(DecoderWrapper::Sequence(sequence)) if sequence.get_decoders().len() == 1 => {
            match &sequence.get_decoders()[0] {
                DecoderWrapper::ByteLevel(decoder) => Some(DecoderWrapper::ByteLevel(*decoder)),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(decoder) = decoder {
        tokenizer.with_decoder(Some(decoder));
    }

    let added_special: Vec<(u32, String)> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, at)| at.special)
        .map(|(id, at)| (id, at.content))
        .collect();

    // The byte-tokenizer dependency uses tokenizers 0.21; serialize the exact
    // checkpoint tokenizer instead of substituting its vocabulary or decoder.
    let serialized = tokenizer.to_string(false).map_err(anyhow::Error::msg)?;
    let mut bt = toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(serialized.as_bytes())?;
    bt.set_eos_token(eos);
    let info = bt.tokrx_info();
    let mut token_bytes = bt.token_bytes();

    for (id, content) in &added_special {
        let idx = *id as usize;
        if idx < token_bytes.len()
            && (token_bytes[idx].is_empty()
                || token_bytes[idx][0] != toktrie::TokTrie::SPECIAL_TOKEN_MARKER)
        {
            let mut bytes = content.as_bytes().to_vec();
            bytes.insert(0, toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
            token_bytes[idx] = bytes;
        }
    }

    let tok_trie = toktrie::TokTrie::from(&info, &token_bytes);
    let env = toktrie_hf_tokenizers::ByteTokenizerEnv {
        tokenizer: bt,
        tok_trie,
    }
    .to_env();
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub fn constraint_from_llg_grammar(
    factory: &ParserFactory,
    grm: TopLevelGrammar,
) -> Result<llguidance::Matcher> {
    let parser = factory.create_parser(grm)?;
    Ok(llguidance::Matcher::new(Ok(parser)))
}
