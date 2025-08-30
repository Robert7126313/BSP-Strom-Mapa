#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    Czech,
}

pub fn tr<'a>(lang: Language, en: &'a str, cz: &'a str) -> &'a str {
    match lang {
        Language::English => en,
        Language::Czech => cz,
    }
}
