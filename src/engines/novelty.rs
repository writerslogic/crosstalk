use std::collections::HashSet;

pub struct NoveltyScorer {
    prior_words: HashSet<String>,
}

impl NoveltyScorer {
    pub fn new() -> Self {
        Self {
            prior_words: HashSet::new(),
        }
    }

    /// Absorb all words from a turn's content into the prior vocabulary.
    pub fn absorb(&mut self, content: &str) {
        for word in tokenize(content) {
            self.prior_words.insert(word);
        }
    }

    /// Score a single sentence: 0.0 = entirely in prior vocabulary, 1.0 = entirely new.
    pub fn sentence_novelty(&self, sentence: &str) -> f64 {
        let words: Vec<String> = tokenize(sentence).collect();
        if words.is_empty() {
            return 0.0;
        }
        let new_count = words
            .iter()
            .filter(|w| !self.prior_words.contains(*w))
            .count();
        new_count as f64 / words.len() as f64
    }

    /// Return the top_k most novel sentences from `content`, with their novelty scores.
    /// Sentences are split on '. ', '! ', '? ', and '\n'.
    pub fn top_novel_sentences(&self, content: &str, top_k: usize) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = split_sentences(content)
            .into_iter()
            .filter(|s| s.split_whitespace().count() >= 5) // skip very short sentences
            .map(|s| {
                let score = self.sentence_novelty(&s);
                (s, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Returns true if `content` is highly novel (mean sentence novelty > 0.4).
    pub fn is_high_novelty(&self, content: &str) -> bool {
        let sentences = split_sentences(content);
        if sentences.is_empty() {
            return false;
        }
        let mean = sentences
            .iter()
            .map(|s| self.sentence_novelty(s))
            .sum::<f64>()
            / sentences.len() as f64;
        mean > 0.4
    }
}

impl Default for NoveltyScorer {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| s.len() >= 3)
        .map(|s| s.to_lowercase())
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') || (ch == '\n' && current.trim().len() > 20) {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}
