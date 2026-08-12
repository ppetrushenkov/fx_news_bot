# Timezone options
TZ_OPTIONS = [
    ("London (UTC+0)", 0),
    ("Frankfurt/Paris (UTC+1)", 1),
    ("Italy (UTC+2)", 2),
    ("Moscow/Istanbul (UTC+3)", 3),
    ("New York (UTC-5)", -5),
    ("Chicago (UTC-6)", -6),
    ("Singapore/Hong Kong (UTC+8)", 8),
]

# ML thresholds for risk settings
ml_thresholds = {
    'conservative': {
        'SFP': 0.8747533895,
        'Extremum Breakout': 0.8475851617,
        'Big Spike': 0.653579109,
        'Chaos': 0.8111326382  # 24h
    },
    'base': {
        'SFP': 0.7818015112,
        'Extremum Breakout': 0.8032437163,
        'Big Spike': 0.5044043359,
        'Chaos': 0.6975647725  # 24h
    },
    'aggressive': {
        'SFP': 0.629072507,
        'Extremum Breakout': 0.5837278663,
        'Big Spike': 0.3903734542,
        'Chaos': 0.4828139359  # 24h
    }
}
