"""Deterministic, text-only broad news topic classification.

Inputs are Headline and Body only. Do not pass source URLs, credibility labels,
publisher names, data_name, or social metadata. Scores are rule evidence points,
not probabilities. Domains are intentionally broad and mutually exclusive.

The caller identifies repeated Body values occurring >= 10 times and passes
ignore_body=True for those rows. They frequently contain unrelated scraped
recommendations in Jruvika. The original combined text need not be altered.
"""

from __future__ import annotations

import html
import re
from typing import Any

CLASSIFIER_VERSION = "keyword-topics-1.1"
BODY_CHARACTER_LIMIT = 2000
TITLE_WEIGHT = 4.0
BODY_WEIGHT = 1.0
BODY_SCORE_CAP = 9.0

DOMAINS = (
    "health", "politics", "entertainment", "conflict", "sports", "business",
    "science_technology", "environment", "society", "other",
)

# Weights express how informative a word/phrase is for the broad topic. A term
# contributes once per field regardless of repetition. Strong terms are worth
# 2 points; supporting terms are worth 1. All matching uses word boundaries.
TERMS: dict[str, dict[int, tuple[str, ...]]] = {
    "health": {
        2: (
            "covid", "covid-19", "coronavirus", "sars-cov-2", "vaccine",
            "vaccines", "vaccination", "cancer", "disease", "diseases",
            "hospital", "hospitals", "medical", "medicine", "medicines",
            "healthcare", "health care", "public health", "mental health",
            "antidepressant", "antidepressants", "antibiotics", "alzheimer",
            "alzheimer's", "dementia", "diabetes", "epidemic", "pandemic",
            "hiv", "aids", "opioid", "opioids", "fentanyl", "pharmaceutical",
            "heart attack", "heart disease", "breast cancer", "blood pressure",
            "obesity", "osteoarthritis", "osteogenesis", "osteoporosis",
            "nutrition", "diet", "diets", "cannabis", "marijuana",
            "birth control", "abortion", "pregnancy", "pregnant", "suicide",
            "autism", "stroke", "depression", "hepatitis", "ebola",
            "drug approval", "drug approvals", "clinical trial", "clinical trials",
            "castration", "fertility", "sterilization", "contraception",
        ),
        1: (
            "health", "healthy", "patient", "patients", "doctor", "doctors",
            "nurse", "nurses", "drug", "drugs", "treatment", "treatments",
            "symptoms", "infection", "infected", "surgery", "surgeries",
            "smoking", "cigarettes", "fitness", "exercise", "painkiller",
            "painkillers", "weight loss", "baby", "babies", "blind",
            "brain", "illness", "illnesses", "poison", "toxic",
        ),
    },
    "politics": {
        2: (
            "president", "presidential", "prime minister", "white house",
            "congress", "congressman", "congresswoman", "senate", "senator",
            "senators", "parliament", "election", "elections", "electoral",
            "republican", "republicans", "democrat", "democrats", "democratic",
            "gop", "trump", "clinton", "obama", "biden", "pence", "corker",
            "tillerson", "mueller", "comey", "sanders", "merkel", "brexit",
            "catalonia", "catalan", "puigdemont", "rajoy", "independence",
            "politics", "political", "politician", "politicians",
            "constitution", "constitutional", "tax reform", "tax bill",
            "supreme court", "cabinet", "diplomat", "diplomats", "diplomatic",
            "state department", "secretary of state", "voter", "voters",
            "voting", "ballot", "campaign", "impeachment", "legislation",
            "legislative", "legislators", "sanctions", "governor", "mayor",
            "putin", "netanyahu", "erdogan", "macron", "theresa may",
        ),
        1: (
            "government", "governments", "federal", "minister", "ministers",
            "policy", "policies", "politically", "conservative", "liberal",
            "liberals", "libertarian", "immigration", "immigrant", "immigrants",
            "senatorial", "referendum", "diplomacy", "democracy", "nra",
            "melania", "first lady", "kremlin", "united nations",
            "coalition government", "lawmaker", "lawmakers", "national anthem",
            "anthem", "gun control", "bump stock", "bump stocks",
        ),
    },
    "entertainment": {
        2: (
            "hollywood", "actor", "actors", "actress", "actresses",
            "celebrity", "celebrities", "movie", "movies", "film", "films",
            "cinema", "box office", "television", "sitcom", "netflix",
            "emmy", "emmys", "oscar", "oscars", "singer", "singers",
            "album", "albums", "rapper", "rap", "music", "musician",
            "musicians", "concert", "song", "songs", "songwriter",
            "weinstein", "eminem", "kardashian", "jenner",
            "kanye", "taylor swift", "selena gomez", "bieber", "beyonce",
            "angelina jolie", "brad pitt", "aniston", "jackie chan",
            "spielberg", "saturday night live", "snl", "s.n.l.", "lorde",
            "jason aldean", "donna karan", "jimmy kimmel", "ellen degeneres",
            "james corden", "lin-manuel miranda", "stephen colbert",
            "harry and meghan", "meghan", "prince harry", "royal wedding",
            "pop star", "pop stars", "rock star", "dance", "dancing",
            "ballet", "theatre", "theater", "art exhibition", "hitchcock",
            "miley cyrus", "george michael", "tom petty", "gossip girl",
            "singing", "hip-hop", "hip hop", "soap opera", "dynasty",
            "novelist", "novel", "booker", "fashion", "fashion designer",
        ),
        1: (
            "entertainment", "showbiz", "tv", "season premiere", "episode",
            "episodes", "documentary", "documentaries", "red carpet",
            "premiere", "comedy", "comedian", "comedienne", "cartoon",
            "literature", "literary", "author", "authors", "poem", "poetry",
            "art", "arts", "artist", "artists", "photography", "photographer",
            "reality show", "screenwriter", "lyrics", "freestyle", "leks",
        ),
    },
    "conflict": {
        2: (
            "syria", "syrian", "assad", "aleppo", "damascus", "raqqa",
            "idlib", "deir ez-zor", "deir ezzor", "mosul", "isis", "isil",
            "islamic state", "taliban", "al qaeda", "al-qaeda", "boko haram",
            "airstrike", "airstrikes", "air strike", "air strikes",
            "nuclear war", "nuclear weapon", "nuclear weapons",
            "nuclear arsenal", "nuclear stockpile", "ballistic missile",
            "missile test", "missile launch", "north korea", "north korean",
            "pyongyang", "kim jong", "chemical attack", "chemical weapons",
            "civil war", "war zone", "war crimes", "war crime", "battlefield",
            "ceasefire", "cease-fire", "shelling", "insurgent", "insurgents",
            "armed forces", "armed conflict", "terrorist", "terrorists",
            "terrorism", "jihadist", "jihadists", "militant", "militants",
            "nato", "pentagon", "combat", "rebels", "rebel-held",
            "palestinian", "palestinians", "gaza", "hezbollah", "hamas",
        ),
        1: (
            "war", "military", "army", "navy", "air force", "troops",
            "soldier", "soldiers", "missile", "missiles", "nuclear",
            "bombing", "bombings", "bomb", "bombs", "artillery", "weapon",
            "weapons", "rebel", "rebellion", "terror", "israeli", "israel",
            "iraq", "iraqi", "afghanistan", "afghan", "yemen", "yemeni",
            "kurd", "kurds", "kurdish", "defense", "defence", "deadly attack",
            "nuclear deal", "arms deal", "iran deal", "iran nuclear",
        ),
    },
    "sports": {
        2: (
            "sport", "sports", "football", "baseball", "basketball",
            "soccer", "hockey", "tennis", "golf", "cricket", "rugby",
            "nfl", "mlb", "nba", "nhl", "ncaa", "fifa", "espn", "ufc",
            "quarterback", "quarterbacks", "touchdown", "touchdowns",
            "pitcher", "pitchers", "home run", "home runs", "innings",
            "strikeouts", "world series", "super bowl", "olympics",
            "olympic", "playoffs", "playoff", "nlds", "alcs", "nlds",
            "yankees", "mets", "pirates", "red sox", "cubs", "orioles",
            "nationals", "indians", "dodgers", "cardinals", "brewers",
            "bucs", "buccaneers", "packers", "raiders", "ravens", "steelers",
            "dolphins", "jets", "browns", "giants", "eagles", "patriots",
            "bills", "vikings", "49ers", "broncos", "chargers", "titans",
            "seahawks", "cowboys", "wolverines", "spartans", "hokies",
            "wake forest", "nc state", "fsu", "notre dame", "liverpool",
            "manchester united", "manchester city", "real madrid", "barcelona",
            "fantasy football", "fantasy baseball", "nld​s", "aaron judge",
            "scherzer", "mccown", "latavius murray", "wayne gallman",
            "doeren", "pitino", "topps", "gregorius", "marlins",
            "world cup", "pga", "atp", "wta", "uefa", "premier league",
            "champions league", "bundesliga", "la liga", "world number one",
            "federer", "nadal", "djokovic", "halep", "beckenbauer", "charlton",
            "serena williams", "venus williams", "ronaldo", "messi",
            "wayne rooney", "southgate", "jetnation", "fanduel", "aussie rules",
            "karate", "martial arts", "man-marking", "game photos",
        ),
        1: (
            "game stories", "game preview", "game review", "film review",
            "expo park", "forbes field", "pnc park", "tr​s-pnc", "trs-pnc",
            "injury report", "field goal", "field goals", "tight end",
            "running back", "wide receiver", "kicker", "box scores",
            "box score", "batting", "pitching", "waiver", "rotoprofessor",
            "coach", "coaches", "tournament", "championship", "champions",
            "athlete", "athletes", "player", "players", "stadium", "stadiums",
            "bowl", "innings", "roster", "draft pick", "racing", "racecar",
            "nascar", "formula one", "formula 1", "f1", "cycling", "cyclist",
            "puckin", "power rankings", "gameday", "opponent preview",
            "wildcard", "2-hitter", "no-hitter", "shutout", "scoreless",
            "game status", "week 4", "week 5", "through the 40s",
            "wr routes", "two-minute drill", "chiefs",
        ),
    },
    "business": {
        2: (
            "economy", "economic", "economics", "inflation", "deflation",
            "recession", "gdp", "stock market", "wall street", "dow jones",
            "nasdaq", "s&p", "ftse", "central bank", "federal reserve",
            "interest rate", "interest rates", "bank bail-out", "bailout",
            "bail-out", "earnings", "quarterly profit", "shareholder",
            "shareholders", "investor", "investors", "investment", "investing",
            "merger", "acquisition", "bankruptcy", "ipo", "bitcoin",
            "cryptocurrency", "cryptocurrencies", "brent",
            "oil prices", "crude oil", "trade deal", "trade talks", "tariff",
            "tariffs", "exports", "imports", "gold price", "bond market",
            "unemployment", "employment data", "payrolls", "sinopec",
            "glencore", "chevron", "bombardier", "wells fargo", "goldman",
            "pemex", "oil field", "oil fields", "easyjet", "air berlin",
            "uber", "ecb", "eurogroup", "evasion", "loan", "loans",
        ),
        1: (
            "business", "businesses", "bank", "banks", "banking", "finance",
            "financial", "money", "wealth", "billionaire", "millionaire",
            "profit", "profits", "revenue", "shares", "stocks", "equity",
            "commodity", "commodities", "trading", "currency", "currencies",
            "trade", "retail", "retailer", "retailers", "ceo", "company",
            "companies", "corporate", "corporation", "firms", "firm",
            "housing market", "real estate", "mortgage", "mortgages", "pension",
            "pensions", "wages", "salary", "salaries", "taxes", "debt",
            "dollar", "euro", "sterling", "opec", "capitalism", "consumer",
            "consumers", "airline", "airlines", "boeing", "airbus",
            "gold", "silver", "shale oil", "oil producer", "oil producers",
        ),
    },
    "science_technology": {
        2: (
            "technology", "technologies", "tech", "software", "hardware",
            "computer", "computers", "computing", "quantum", "robot",
            "robots", "robotics", "artificial intelligence", "machine learning",
            "internet", "smartphone", "smartphones", "iphone", "android",
            "google", "facebook", "apple", "microsoft", "samsung", "amazon",
            "cyber", "cybersecurity", "cyberattack", "cyberattacks", "hacking",
            "hacker", "hackers", "encryption", "smart meter", "smart meters",
            "nasa", "spacex", "astronaut", "astronauts", "spacecraft",
            "space station", "rocket", "rockets", "satellite", "satellites",
            "astronomy", "astrophysics", "telescope", "telescopes", "mars",
            "galaxy", "galaxies", "planet", "planets", "planet x", "nibiru",
            "ufo", "ufos", "extraterrestrial", "extraterrestrials", "aliens",
            "flying car", "flying cars", "electric car", "driverless",
            "self-driving", "tesla", "lithium", "dna", "gene editing",
            "science", "scientific", "physics", "physicist", "chemistry",
            "scientist", "scientists", "haarp", "aurora", "potato battery",
            "cellphone", "cellphones", "license plate tracking", "microwave",
            "periodic table", "super elements", "innovation", "innovations",
        ),
        1: (
            "researchers", "research", "study", "innovation", "innovations",
            "discovery", "discoveries", "digital", "online", "wireless",
            "camera", "cameras", "gadget", "gadgets", "electronics",
            "surveillance", "privacy", "data breach", "data breaches",
            "bot", "bots", "algorithm", "algorithms", "simulation",
            "virtual reality", "virtual", "app", "apps", "battery",
            "batteries", "solar panel", "solar panels", "3d printing",
            "strange object", "sky phenomena", "triangle shadow",
        ),
    },
    "environment": {
        2: (
            "wildfire", "wildfires", "forest fire", "forest fires",
            "hurricane", "hurricanes", "earthquake", "earthquakes",
            "tsunami", "tsunamis", "tornado", "tornadoes", "cyclone",
            "flood", "floods", "flooding", "drought", "volcano", "volcanic",
            "climate", "climate change", "global warming", "greenhouse",
            "pollution", "polluted", "deforestation", "biodiversity",
            "conservation", "endangered", "extinction", "arctic", "antarctic",
            "environment", "environmental", "wildlife", "ecosystem",
            "ecosystems", "renewable", "renewables", "paris accord",
            "paris agreement", "solar storm", "geomagnetic", "sinkhole",
        ),
        1: (
            "weather", "storm", "storms", "disaster", "disasters", "evacuation",
            "evacuations", "evacuated", "evacuate", "firefighters", "firefighter",
            "heavy rain", "heat wave", "heatwave", "blizzard", "snowstorm",
            "ocean", "oceans", "coral", "sea level", "carbon emissions",
            "carbon dioxide", "recycling", "recycle", "sustainability",
            "sustainable", "wind farm", "wind farms", "fire events",
            "hurricane maria", "hurricane irma", "hurricane harvey",
        ),
    },
    "society": {
        2: (
            "shooting", "shootings", "gunman", "gunmen", "massacre",
            "murder", "murdered", "homicide", "serial killer", "kidnapping",
            "slaughter", "shooter", "shooters", "gunshot", "gunshots",
            "kidnapped", "rape", "rapist", "rapists", "sexual assault",
            "sexual harassment", "sexual abuse", "molestation", "burglary",
            "robbery", "police", "prison", "prisons", "prisoner", "prisoners",
            "jail", "inmate", "inmates", "arrest", "arrested", "arrests",
            "court", "trial", "judge", "jury", "lawsuit", "prosecutor",
            "prosecutors", "criminal", "crime", "crimes", "pedophile",
            "paedophile", "racist", "racism", "white supremacist",
            "refugee", "refugees", "detention", "deportation", "asylum",
            "vatican", "pope", "priest", "priests", "church", "churches",
            "religion", "religious", "bible", "christian", "christians",
            "muslim", "muslims", "mosque", "jewish", "judaism", "islam",
            "school", "schools", "student", "students", "teacher", "teachers",
            "university", "universities", "education", "college", "colleges",
            "homeless", "homelessness", "poverty", "human rights",
            "sentenced", "praying", "parental rights", "church service",
            "tourism", "tourist", "tourists", "holiday destinations",
            "pyramids", "mona lisa", "museum", "museums", "archaeology",
        ),
        1: (
            "society", "social", "community", "communities", "family",
            "families", "children", "child", "kids", "women", "marriage",
            "married", "wedding", "divorce", "relationship", "relationships",
            "romance", "dating", "cat", "cats", "dog", "dogs", "pets", "pet",
            "animal", "animals", "travel", "tourism", "tourist", "tourists",
            "charity", "charities", "charitable", "fundraiser", "fundraising",
            "volunteer", "volunteers", "protest", "protests", "protester",
            "protesters", "scandal", "immigrant", "immigrants", "immigration",
            "accident", "crash", "crashes", "traffic", "road", "roads",
            "missing", "rescued", "rescue", "victim", "victims", "violence",
            "violent", "fire", "fires", "gun", "guns", "gunfire", "warrant",
            "warrantless", "nazi", "nazis", "fascist", "fascism",
            "church", "parent", "parents", "overtime",
            "monument", "monuments", "heritage", "history", "historical",
            "ancient", "recipe", "recipes", "chocolate", "cooking", "food",
            "breakfast", "lunch", "dinner", "chef", "chefs", "restaurant",
        ),
    },
}


def _normalize(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]*>", " ", value)
    # Strip URL strings embedded in Body too, not just the original URLs field.
    value = re.sub(r"(?:https?://|www\.)\S+", " ", value, flags=re.I)
    value = value.translate(str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", value).strip().lower()


# For exact ties, prefer the more specific topic over generic political names.
# This covers, for example, a rapper mentioning a president in a music story.
_TIE_PRIORITY = (
    "sports", "entertainment", "health", "conflict", "environment",
    "science_technology", "business", "society", "politics",
)

_COMPILED: dict[str, list[tuple[str, float, str, re.Pattern[str]]]] = {}
for _domain, _weighted_terms in TERMS.items():
    _COMPILED[_domain] = []
    _seen: set[str] = set()
    for _weight, _terms in _weighted_terms.items():
        for _term in _terms:
            _term = _normalize(_term).replace("\u200b", "")
            if _term in _seen:
                continue
            _seen.add(_term)
            _pattern = re.compile(r"(?<!\w)" + re.escape(_term).replace(r"\ ", r"\s+") + r"(?!\w)")
            _first_token = re.findall(r"\w+", _term)[0]
            _COMPILED[_domain].append((_term, float(_weight), _first_token, _pattern))


def _field_scores(value: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}
    tokens = set(re.findall(r"\w+", value))
    for domain, patterns in _COMPILED.items():
        hits = [(term, weight) for term, weight, first, pattern in patterns
                if first in tokens and pattern.search(value)]
        # Do not double-count a short term contained within a matched phrase:
        # "health" + "public health", "nuclear" + "nuclear war", etc.
        maximal = [(term, weight) for term, weight in hits if not any(
            term != longer and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", longer)
            for longer, _ in hits
        )]
        scores[domain] = sum(weight for _, weight in maximal)
        matches[domain] = [term for term, _ in maximal]
    return scores, matches


def _top(scores: dict[str, float]) -> str:
    return max(scores, key=lambda domain: (scores[domain], -_TIE_PRIORITY.index(domain)))


def classify(headline: str, body: str, ignore_body: bool = False) -> dict[str, Any]:
    """Return a deterministic broad topic and transparent evidence metadata.

    score/runner_up_score/margin are evidence points, never probabilities.
    needs_review signals weak, ambiguous, or contradictory rule evidence; a
    False value does not imply human verification or establish accuracy.
    """
    title_text = _normalize(headline)
    # Truncate raw Body before normalization so resource use is bounded.
    body_text = "" if ignore_body else _normalize((body or "")[:BODY_CHARACTER_LIMIT])
    title_scores, title_matches = _field_scores(title_text)
    body_scores, body_matches = _field_scores(body_text)
    title_top = _top(title_scores)
    body_top = _top(body_scores)
    title_max = title_scores[title_top]
    body_max = body_scores[body_top]

    scores = {
        domain: TITLE_WEIGHT * title_scores[domain] + BODY_WEIGHT * min(BODY_SCORE_CAP, body_scores[domain])
        for domain in title_scores
    }

    # An informative title receives priority over an unrelated long scraped
    # Body. Body still breaks title ties and contributes auditable evidence.
    strong_title = title_max >= 2
    if strong_title:
        eligible = {domain for domain, value in title_scores.items() if value >= title_max}
        top_domain = max(eligible, key=lambda domain: (scores[domain], -_TIE_PRIORITY.index(domain)))
    else:
        top_domain = _top(scores)
    alternatives = [domain for domain in scores if domain != top_domain]
    runner = max(alternatives, key=lambda domain: (scores[domain], -_TIE_PRIORITY.index(domain)))
    top_score = scores[top_domain]
    runner_score = scores[runner]
    # The chosen title-priority domain can have fewer raw points than another
    # category. Preserve the actual signed margin for transparent review.
    margin = top_score - runner_score
    disagreement = bool(title_max >= 2 and body_max >= 3 and top_domain != body_top
                        and body_max - body_scores[top_domain] >= 2)
    domain = top_domain if top_score >= 3 else "other"
    needs_review = bool(
        domain == "other" or top_score < 8 or margin < 4 or disagreement
        or (not title_text and not body_text)
    )
    return {
        "domain": domain,
        "score": round(top_score, 3),
        "runner_up": runner,
        "runner_up_score": round(runner_score, 3),
        "margin": round(margin, 3),
        "needs_review": needs_review,
        "matched_terms": {
            "headline": title_matches[top_domain],
            "body": body_matches[top_domain],
        },
        "runner_up_matched_terms": {
            "headline": title_matches[runner],
            "body": body_matches[runner],
        },
        "title_body_disagreement": disagreement,
        "title_domain": top_domain if strong_title else (title_top if title_max else "other"),
        "body_domain": body_top if body_max else "other",
        "body_ignored": bool(ignore_body),
        "body_character_limit": BODY_CHARACTER_LIMIT,
        "classifier_version": CLASSIFIER_VERSION,
        "score_type": "keyword_evidence_points_not_probability",
    }

