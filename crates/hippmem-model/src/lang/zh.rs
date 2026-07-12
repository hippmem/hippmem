//! Chinese (zh) locale data for HIPPMEM's deterministic NLP pipeline.
//!
//! Contains ALL Chinese-specific patterns: question detection keywords,
//! explanatory markers, content markers (goal/event/emotion/decision/causal),
//! stop words, and query processing rules.
//!
//! Tokenization for zh uses jieba (see `hippmem-core::hash::tokenize`).

use super::{EmotionKind, LangData};

pub const ZH: LangData = LangData {
    locale: "zh",

    // ── Question type detection ──
    q_correction: &[
        "修正",
        "纠正",
        "改过",
        "反悔",
        "变化",
        "改变",
        "调整",
        "之前说",
    ],
    q_preference: &["偏好", "喜欢", "习惯", "倾向", "看重", "重视", "最喜欢"],
    q_why: &["为什么", "为何", "为啥"],
    q_how: &["怎么", "如何", "怎样"],
    q_what: &["什么", "哪些"],

    // ── Explanatory / causal scoring ──
    explanatory: &[
        ("因为", 0.12),
        ("决定", 0.10),
        ("原因", 0.15),
        ("由于", 0.15),
        ("所以", 0.10),
        ("因此", 0.10),
        ("导致", 0.12),
    ],

    // ── Content marker extraction ──
    goal_markers: &["目标", "打算", "计划", "想要"],
    event_markers: &["完成", "部署", "上线", "发布", "开会", "讨论"],
    decision_markers: &["决定", "选用", "采用", "弃用"],
    preference_pos: &["喜欢", "倾向"],
    preference_neg: &["讨厌", "避免"],
    causal_pairs: &[("因为", "所以"), ("由于", "导致"), ("因为", "因此")],
    emotion_keywords: &[
        ("开心", EmotionKind::Joy),
        ("兴奋", EmotionKind::Joy),
        ("满意", EmotionKind::Joy),
        ("沮丧", EmotionKind::Frustration),
        ("失望", EmotionKind::Frustration),
        ("焦虑", EmotionKind::Anxiety),
        ("生气", EmotionKind::Anger),
        ("害怕", EmotionKind::Fear),
    ],

    // ── Query pre-processing ──
    stop_words: &[
        "什么",
        "怎么",
        "如何",
        "为什么",
        "哪些",
        "哪个",
        "的",
        "了",
        "是",
        "有",
        "在",
        "和",
        "与",
        "或",
        "对",
        "把",
        "被",
        "让",
        "从",
        "到",
        "中",
        "上",
        "下",
        "这",
        "那",
        "吗",
        "呢",
        "啊",
        "吧",
        "问题",
        "说法",
        "关于",
        "一个",
        "一些",
        "可以",
        "没有",
        "不是",
        "他们",
        "我们",
    ],
    what_delimiters: &["是什么", "是指什么", "指的是什么"],
    definition_patterns: &["是", "使用", "基于", "采用"],
    possessive_particle: Some('的'),
    change_pair: Some(("之前", "后来")),
};
