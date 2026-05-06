MONETARY_POLICY = [
    "rate", "policy", "monetary", "fomc", "mpc", "meeting",
    "minutes", "statement", "votes",
    "asset purchase", "purchase facility", "refinancing",
    "cash rate", "bank rate", "official cash",
    "fed", "ecb", "boe", "boj", "rba", "rbnz", "boc", "snb",
    "overnight rate", "federal funds", "funds rate"
]

CB_SPEECH = [
    "speech", "speaks", "press conference", 
    "chair", "president", "gov", "press",
    "powell", "lagarde", "bailey", "kuroda", "draghi",
    "yellen", "carney", "bullard", "waller", "kashkari",
    "member"
]

INFLATION = [
    "cpi", "core cpi", "ppi", "core ppi",
    "pce", "core pce",
    "inflation", "price index", "prices",
    "median cpi", "trimmed cpi",
    "inflation expectations"
]

LABOR_MARKET = [
    "employment", "unemployment", "job", "claims",
    "non farm", "nonfarm", "payrolls",
    "adp", "jolts", "job openings",
    "earnings", "hourly earnings",
    "job cuts", "challenger"
]

ECONOMIC_ACTIVITY = [
    "gdp", "industrial", "production",
    "manufacturing", "services",
    "orders", "factory orders",
    "output", "capacity", "utilization",
    "productivity", "investment"
]

SENTIMENT = [
    "pmi", "ism", "confidence", "sentiment",
    "ifo", "zew", "gfk", "nfib", "philly",
    "survey", "optimism", "barometer",
    "expectations", "watchers"
]

CONSUMER_HOUSING = [
    "retail", "sales", "consumer",
    "housing", "home", "mortgage",
    "building", "permits", "starts",
    "house price", "hpi"
]

TRADE_FINANCE = [
    "trade", "balance", "current account",
    "budget", "borrowing", "credit",
    "money supply", "m2", "m3",
    "lending", "loans", "reserves",
    "foreign", "currency"
]

COMMODITIES = [
    "oil", "crude", "gas",
    "inventories", "storage",
    "opec", "commodity prices"
]


CLASS_KEYWORDS = {
    "MONETARY_POLICY": MONETARY_POLICY,
    "CB_SPEECH": CB_SPEECH,
    "INFLATION": INFLATION,
    "LABOR_MARKET": LABOR_MARKET,
    "ECONOMIC_ACTIVITY": ECONOMIC_ACTIVITY,
    "SENTIMENT": SENTIMENT,
    "CONSUMER_HOUSING": CONSUMER_HOUSING,
    "TRADE_FINANCE": TRADE_FINANCE,
    "COMMODITIES": COMMODITIES
}


EVENT_WEIGHTS_D = {
    'Interest_Rate_Decision': 5,
    'FOMC': 5,
    'Inflation_rate': 5,
    'Core_Inflation_rate': 5,
    'NFP': 4,
    'GDP': 4,
    'Unemployment_rate': 4,
    'PMI': 3,
    'PMI_Manufacturing': 3,
    'PMI_Services': 3,
    'Retail_Sales': 3,
    'Balance_of_Trade': 2
}