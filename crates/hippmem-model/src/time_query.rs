//! Temporal query parsing for time-aware retrieval (proposal
//! `query-time-aware-retrieval`, confirmed 2026-08-27).
//!
//! Pure rules — never goes through the extractor, so hash and neural
//! backends behave identically. Locale literals live in `lang/<locale>.rs`
//! (P6); the numeric/date logic lives here.

use crate::lang::{active_locales, TimeTermKind};
use hippmem_core::time::{civil_from_days, days_from_civil, Timestamp};

/// Parsed temporal intent of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalQuerySpec {
    /// A single target day (as a UTC day count). Retrieval covers the
    /// [day-1, day, day+1] buckets — local dates straddle the UTC day
    /// boundary, and the neighbours absorb the offset.
    SingleDay { day: i64 },
    /// Inclusive day range (UTC day counts).
    Range { start_day: i64, end_day: i64 },
}

/// Cap on the number of day buckets enumerated for a range query (one
/// quarter ≈ 93 days).
pub const RANGE_MAX_DAYS: i64 = 93;

const MS_PER_DAY: i64 = 86_400_000;

fn day_of(ts: Timestamp) -> i64 {
    ts.0 / MS_PER_DAY
}

/// Parses a query for temporal intent. Returns `None` when no temporal
/// expression is present (or it is too ambiguous to trust) — the caller
/// then falls back to the current-time buckets.
pub fn parse_temporal_query(query: &str, now: Timestamp) -> Option<TemporalQuerySpec> {
    let today = day_of(now);
    let (cy, _cm, _cd) = civil_from_days(today);

    // 1) Ranges first: "3月到5月" / "3月5日到3月10日" / "March to May"
    //    must not be eaten by the absolute parser (which would read
    //    "3月到5月" as "3月1日"). A recognized-but-too-large range is a
    //    conservative None — it must not degrade to a single day either.
    match parse_zh_range(query, cy) {
        RangeParse::Spec(spec) => return Some(spec),
        RangeParse::TooLarge => return None,
        RangeParse::NotRange => {}
    }
    match parse_en_range(query, cy) {
        RangeParse::Spec(spec) => return Some(spec),
        RangeParse::TooLarge => return None,
        RangeParse::NotRange => {}
    }

    // 2) Absolute dates: zh ("2026年3月5日" / "3月5日"), ISO (2026-03-05),
    //    en ("March 5, 2026").
    if let Some((y, m, d)) = parse_zh_absolute(query, today) {
        return Some(TemporalQuerySpec::SingleDay {
            day: days_from_civil(y, m, d),
        });
    }
    if let Some((y, m, d)) = parse_iso_absolute(query) {
        return Some(TemporalQuerySpec::SingleDay {
            day: days_from_civil(y, m, d),
        });
    }
    if let Some((y, m, d)) = parse_en_absolute(query, cy) {
        return Some(TemporalQuerySpec::SingleDay {
            day: days_from_civil(y, m, d),
        });
    }

    // 3) Relative terms from the locale tables ("昨天" / "last week" / ...).
    match_term(query).map(|kind| spec_for_term(kind, today))
}

/// Parses "YYYY年M月D日" (year optional) and "M月D日" (current year;
/// a date in the future rolls back to last year — "12月31日" asked on
/// Jan 1 means last year).
fn parse_zh_absolute(q: &str, today: i64) -> Option<(i64, u32, u32)> {
    let (year, start) = match q.find('年') {
        Some(pos) => {
            let y: i64 = q[..pos].trim().parse().ok()?;
            if !(1970..=2100).contains(&y) {
                return None;
            }
            (y, pos + '年'.len_utf8())
        }
        None => (0, 0),
    };
    let mpos = q[start..].find('月')? + start;
    let m: u32 = q[start..mpos].trim().parse().ok()?;
    if !(1..=12).contains(&m) {
        return None;
    }
    let after_m = mpos + '月'.len_utf8();
    let day = match q[after_m..].find('日') {
        Some(dpos) => {
            let d: u32 = q[after_m..after_m + dpos].trim().parse().ok()?;
            if !(1..=31).contains(&d) {
                return None;
            }
            d
        }
        None => 1,
    };
    // Yearless dates: a target in the future (beyond today) means last year
    // ("12月31日" asked on Jan 1 refers to the previous year).
    let y = if year != 0 {
        year
    } else {
        let d = days_from_civil(today_year(today), m, day);
        if d > today {
            today_year(today) - 1
        } else {
            today_year(today)
        }
    };
    Some((y, m, day))
}

fn today_year(today: i64) -> i64 {
    civil_from_days(today).0
}

/// Parses "YYYY-MM-DD" / "YYYY-M-D". Char-boundary safe (queries may be
/// CJK — byte slicing would split multibyte characters).
fn parse_iso_absolute(q: &str) -> Option<(i64, u32, u32)> {
    let chars: Vec<(usize, char)> = q.char_indices().collect();
    for i in 0..chars.len().saturating_sub(9) {
        // Candidate: 4 digits, '-', 1-2 digits, '-', 1-2 digits.
        let tail: String = chars[i..i + 4].iter().map(|(_, c)| *c).collect();
        let Ok(y) = tail.parse::<i64>() else {
            continue;
        };
        if chars.get(i + 4).map(|(_, c)| *c) != Some('-') {
            continue;
        }
        let Some(m_end) = chars[i + 5..].iter().position(|(_, c)| *c == '-') else {
            continue;
        };
        let m: String = chars[i + 5..i + 5 + m_end]
            .iter()
            .map(|(_, c)| *c)
            .collect();
        let Ok(m) = m.parse::<u32>() else {
            continue;
        };
        let d: String = chars[i + 6 + m_end..]
            .iter()
            .take_while(|(_, c)| c.is_ascii_digit())
            .map(|(_, c)| *c)
            .take(2)
            .collect();
        if d.is_empty() {
            continue;
        }
        let Ok(d) = d.parse::<u32>() else {
            continue;
        };
        if (1..=12).contains(&m) && (1..=31).contains(&d) && (1970..=2100).contains(&y) {
            return Some((y, m, d));
        }
    }
    None
}

/// Parses "March 5" / "March 5, 2026" (English month names).
fn parse_en_absolute(q: &str, cy: i64) -> Option<(i64, u32, u32)> {
    const MONTHS: [(&str, u32); 12] = [
        ("January", 1),
        ("February", 2),
        ("March", 3),
        ("April", 4),
        ("May", 5),
        ("June", 6),
        ("July", 7),
        ("August", 8),
        ("September", 9),
        ("October", 10),
        ("November", 11),
        ("December", 12),
    ];
    for (name, mnum) in MONTHS {
        if let Some(pos) = q.find(name) {
            let rest = &q[pos + name.len()..];
            let rest = rest.trim_start();
            // "March" alone is not a day — the caller treats it as a range
            // only in "March to May" (handled elsewhere).
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                continue;
            }
            let d: u32 = digits.parse().ok()?;
            if !(1..=31).contains(&d) {
                return None;
            }
            let after = rest[digits.len()..].trim_start();
            let y = match after.find(',') {
                Some(cpos) => {
                    let ystr: String = after[cpos + 1..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if ystr.is_empty() {
                        cy
                    } else {
                        ystr.parse().ok()?
                    }
                }
                None => cy,
            };
            if !(1970..=2100).contains(&y) {
                return None;
            }
            return Some((y, mnum, d));
        }
    }
    None
}

/// Result of trying to parse a range expression: distinguishes "no range
/// expression here" from "range expression found but too large / invalid"
/// — the latter must NOT fall through to the absolute parser (which would
/// eat "2020年1月到2026年3月" as the single day 2020-01-01).
enum RangeParse {
    NotRange,
    TooLarge,
    Spec(TemporalQuerySpec),
}

/// Parses "M月D日到M月D日" (current year) and month ranges, which may
/// carry explicit years: "M月到M月" (current year), "YYYY年M月到YYYY年M月",
/// "YYYY年M月到M月" (right inherits the left year).
fn parse_zh_range(q: &str, cy: i64) -> RangeParse {
    let dao = match q.find("到") {
        Some(d) => d,
        None => return RangeParse::NotRange,
    };
    let left = &q[..dao];
    let right = &q[dao + '到'.len_utf8()..];

    // "M月D日到M月D日"
    if let (Some((m1, d1)), Some((m2, d2))) = (parse_month_day(left), parse_month_day(right)) {
        if m2 < m1 {
            return RangeParse::NotRange; // wrap-around years unsupported in v1
        }
        return range_parse(days_from_civil(cy, m1, d1), days_from_civil(cy, m2, d2));
    }
    // "M月到M月" / "YYYY年M月到YYYY年M月" / "YYYY年M月到M月"
    if let (Some((y1, m1)), Some((y2, m2))) = (parse_year_month(left), parse_year_month(right)) {
        let y1 = y1.unwrap_or(cy);
        let y2 = y2.unwrap_or(y1);
        if y2 < y1 || (y2 == y1 && m2 < m1) {
            return RangeParse::NotRange;
        }
        return range_parse(
            days_from_civil(y1, m1, 1),
            days_from_civil(y2, m2, last_day_of_month(y2, m2)),
        );
    }
    RangeParse::NotRange
}

fn range_parse(start_day: i64, end_day: i64) -> RangeParse {
    if end_day < start_day || end_day - start_day + 1 > RANGE_MAX_DAYS {
        RangeParse::TooLarge
    } else {
        RangeParse::Spec(TemporalQuerySpec::Range { start_day, end_day })
    }
}

/// "M月" or "YYYY年M月"; returns (optional year, month).
fn parse_year_month(s: &str) -> Option<(Option<i64>, u32)> {
    match s.find('年') {
        Some(yp) => {
            let y: i64 = s[..yp].trim().parse().ok()?;
            if !(1970..=2100).contains(&y) {
                return None;
            }
            let m = parse_month_only(&s[yp + '年'.len_utf8()..])?;
            Some((Some(y), m))
        }
        None => {
            let m = parse_month_only(s)?;
            Some((None, m))
        }
    }
}

/// Parses "March to May" (English month-name range, current year).
fn parse_en_range(q: &str, cy: i64) -> RangeParse {
    const MONTHS: [(&str, u32); 12] = [
        ("January", 1),
        ("February", 2),
        ("March", 3),
        ("April", 4),
        ("May", 5),
        ("June", 6),
        ("July", 7),
        ("August", 8),
        ("September", 9),
        ("October", 10),
        ("November", 11),
        ("December", 12),
    ];
    let to = match q.find(" to ") {
        Some(t) => t,
        None => return RangeParse::NotRange,
    };
    let left = &q[..to];
    let right = &q[to + 4..];
    let Some((_, m1)) = MONTHS
        .iter()
        .find(|(name, _)| left.trim_end().ends_with(name))
    else {
        return RangeParse::NotRange;
    };
    let Some((_, m2)) = MONTHS
        .iter()
        .find(|(name, _)| right.trim_start().starts_with(name))
    else {
        return RangeParse::NotRange;
    };
    if m2 < m1 {
        return RangeParse::NotRange;
    }
    range_parse(
        days_from_civil(cy, *m1, 1),
        days_from_civil(cy, *m2, last_day_of_month(cy, *m2)),
    )
}

/// "M月D日" within a larger string.
fn parse_month_day(s: &str) -> Option<(u32, u32)> {
    let mpos = s.find('月')?;
    let m: u32 = s[..mpos].trim().parse().ok()?;
    if !(1..=12).contains(&m) {
        return None;
    }
    let after_m = &s[mpos + '月'.len_utf8()..];
    let dpos = after_m.find('日')?;
    let d: u32 = after_m[..dpos].trim().parse().ok()?;
    if !(1..=31).contains(&d) {
        return None;
    }
    Some((m, d))
}

/// "M月" within a larger string.
fn parse_month_only(s: &str) -> Option<u32> {
    let mpos = s.find('月')?;
    let m: u32 = s[..mpos].trim().parse().ok()?;
    if (1..=12).contains(&m) {
        Some(m)
    } else {
        None
    }
}

fn last_day_of_month(year: i64, month: u32) -> u32 {
    let (ny, nm) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    (days_from_civil(ny, nm, 1) - days_from_civil(year, month, 1)) as u32
}

/// Matches locale time terms (longest first — "上个月" contains "上月").
fn match_term(query: &str) -> Option<TimeTermKind> {
    let mut terms: Vec<(&str, TimeTermKind)> = Vec::new();
    for locale in active_locales() {
        for (term, kind) in locale.time_terms {
            terms.push((term, *kind));
        }
    }
    terms.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
    for (term, kind) in terms {
        if query.contains(term) {
            return Some(kind);
        }
    }
    None
}

/// Maps a matched term to a day/range spec relative to today.
fn spec_for_term(kind: TimeTermKind, today: i64) -> TemporalQuerySpec {
    use TimeTermKind::*;
    match kind {
        Today => TemporalQuerySpec::SingleDay { day: today },
        Yesterday => TemporalQuerySpec::SingleDay { day: today - 1 },
        DayBeforeYesterday => TemporalQuerySpec::SingleDay { day: today - 2 },
        ThisWeek => {
            let (y, m, d) = civil_from_days(today);
            let dow = weekday(today);
            let monday = today - dow as i64;
            let _ = (y, m, d);
            TemporalQuerySpec::Range {
                start_day: monday,
                end_day: monday + 6,
            }
        }
        LastWeek => {
            let monday = today - weekday(today) as i64;
            TemporalQuerySpec::Range {
                start_day: monday - 7,
                end_day: monday - 1,
            }
        }
        ThisMonth => {
            let (y, m, _) = civil_from_days(today);
            TemporalQuerySpec::Range {
                start_day: days_from_civil(y, m, 1),
                end_day: days_from_civil(y, m, last_day_of_month(y, m)),
            }
        }
        LastMonth => {
            let (y, m, _) = civil_from_days(today);
            let (ly, lm) = if m == 1 { (y - 1, 12) } else { (y, m - 1) };
            TemporalQuerySpec::Range {
                start_day: days_from_civil(ly, lm, 1),
                end_day: days_from_civil(ly, lm, last_day_of_month(ly, lm)),
            }
        }
        ThisYear => {
            let (y, _, _) = civil_from_days(today);
            TemporalQuerySpec::Range {
                start_day: days_from_civil(y, 1, 1),
                end_day: days_from_civil(y, 12, 31),
            }
        }
    }
}

/// Day of week for a day count: 0 = Monday .. 6 = Sunday.
fn weekday(days: i64) -> u32 {
    // 1970-01-01 was a Thursday; days_from_civil(1970,1,1)=0.
    ((days + 3).rem_euclid(7)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_772_640_000_000; // 2026-03-05T00:00:00Z

    fn now() -> Timestamp {
        Timestamp(NOW)
    }

    #[test]
    fn zh_absolute_with_year() {
        let spec = parse_temporal_query("2026年3月5日做了什么", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::SingleDay {
                day: days_from_civil(2026, 3, 5)
            }
        );
    }

    #[test]
    fn zh_absolute_yearless_rolls_back_future() {
        // Asked on 2026-03-05, "12月31日" cannot be 2026 — roll back to 2025.
        let spec = parse_temporal_query("12月31日做了什么", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::SingleDay {
                day: days_from_civil(2025, 12, 31)
            }
        );
    }

    #[test]
    fn iso_absolute() {
        let spec = parse_temporal_query("2026-03-05的任务", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::SingleDay {
                day: days_from_civil(2026, 3, 5)
            }
        );
    }

    #[test]
    fn en_absolute() {
        let spec = parse_temporal_query("what did I do on March 5, 2026", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::SingleDay {
                day: days_from_civil(2026, 3, 5)
            }
        );
    }

    #[test]
    fn relative_terms() {
        let t = day_of(now());
        assert_eq!(
            parse_temporal_query("昨天做了什么", now()),
            Some(TemporalQuerySpec::SingleDay { day: t - 1 })
        );
        assert_eq!(
            parse_temporal_query("前天", now()),
            Some(TemporalQuerySpec::SingleDay { day: t - 2 })
        );
        assert!(matches!(
            parse_temporal_query("上周", now()),
            Some(TemporalQuerySpec::Range { .. })
        ));
        assert!(matches!(
            parse_temporal_query("上个月", now()),
            Some(TemporalQuerySpec::Range { .. })
        ));
        assert!(matches!(
            parse_temporal_query("last week", now()),
            Some(TemporalQuerySpec::Range { .. })
        ));
    }

    #[test]
    fn zh_month_range() {
        let spec = parse_temporal_query("3月到5月做了哪些事", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::Range {
                start_day: days_from_civil(2026, 3, 1),
                end_day: days_from_civil(2026, 5, 31),
            }
        );
    }

    #[test]
    fn zh_day_range() {
        let spec = parse_temporal_query("3月5日到3月10日", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::Range {
                start_day: days_from_civil(2026, 3, 5),
                end_day: days_from_civil(2026, 3, 10),
            }
        );
    }

    #[test]
    fn en_month_range() {
        let spec = parse_temporal_query("from March to May", now()).unwrap();
        assert_eq!(
            spec,
            TemporalQuerySpec::Range {
                start_day: days_from_civil(2026, 3, 1),
                end_day: days_from_civil(2026, 5, 31),
            }
        );
    }

    #[test]
    fn no_temporal_expression() {
        assert_eq!(parse_temporal_query("小明住在哪里", now()), None);
        assert_eq!(parse_temporal_query("what is go lang", now()), None);
    }

    #[test]
    fn range_capped() {
        // 2020年1月到2026年3月 exceeds the cap → None (conservative).
        assert_eq!(parse_temporal_query("2020年1月到2026年3月", now()), None);
    }

    #[test]
    fn civil_roundtrip() {
        for (y, m, d) in [(2026, 3, 5), (1970, 1, 1), (2024, 2, 29), (2099, 12, 31)] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m, d));
        }
    }
}
