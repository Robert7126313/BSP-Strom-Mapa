//! Minimal two-language helper (English and Czech).

/// The supported UI languages.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    Czech,
}

/// Translates a string based on the selected language.
///
/// This function takes an English string and a Czech string and returns the
/// appropriate one based on the `lang` parameter.
pub fn tr<'a>(lang: Language, en: &'a str, cz: &'a str) -> &'a str {
    match lang {
        Language::English => en,
        Language::Czech => cz,
    }
}
