# Backtester Engine — Customer Archetypes & Go-to-Market Analysis

---

## 1. Product Interpretation

**What it is:** A Rust-core, Python-wrapped backtesting engine for quantitative trading research. It competes in the same space as vectorbt Pro, Backtrader, and Zipline but differentiates on three axes:

| Axis | Value |
|------|-------|
| **Speed** | 5–100x faster than pure-Python alternatives. 1000-param sweeps in <30s. Sub-200ms single backtests on 1yr of 1-second bars. |
| **Realism** | Volume-impact slippage, partial fills, perpetual funding rates, borrow costs, maker/taker fees, limit orders, trailing stops. |
| **Research tooling** | Walk-forward optimization, parameter stability analysis, 2D heatmaps, Monte Carlo, bit-for-bit replay via manifests. |

**The real problem it solves:**
Quantitative researchers lose days or weeks to (a) slow iteration loops when sweeping parameters, (b) overfitting because they lack walk-forward and stability tools, and (c) unrealistic PnL because their backtest ignores fees, slippage, and market impact. This engine compresses the research cycle from weeks to hours while producing PnL estimates that survive contact with live markets.

**When the product becomes worth paying for:**
The moment a user's research loop is bottlenecked by compute time, or the moment they deploy a strategy live and the backtest PnL diverges materially from realized PnL due to naive execution modeling.

---

## 2. Ranked Target Segments

| Rank | Segment | Likelihood to Pay | Budget | Urgency | Market Size | Accessibility |
|------|---------|-------------------|--------|---------|-------------|---------------|
| 1 | Crypto-native quant traders (semi-pro / full-time) | **High** | $500–3,000/yr | High | ~50,000 globally | High (Twitter/X, Discord, Telegram) |
| 2 | Micro-teams crypto (2–5 traders pooling capital) | **Medium-High** | $1,000–5,000/yr | Medium-High | ~15,000–30,000 globally | Medium (Discord, Telegram, Twitter/X) |
| 3 | Signal sellers & strategy-as-a-service operators | **High** | $1,000–5,000/yr | Medium-High | ~10,000–20,000 globally | High (Twitter/X, trading forums) |
| 4 | Quantitative finance students & bootcamp grads | **Medium** | $0–200/yr | Medium | ~200,000+ | Very High (Reddit, YouTube, university) |
| 5 | Petites fintech / outils de trading (embedded engine) | **Medium** | $5,000–20,000/yr | Low-Medium | ~500–1,000 startups | Low (outbound, GitHub) |
| 6 | Retail algo traders (hobby) | **Low** | $0–100/yr | Low | ~500,000+ | Very High but low conversion |

---

## 3. Detailed Customer Archetypes

---

### ARCHETYPE 1: "The Crypto Quant" — Semi-Professional Crypto Trader

**User type:** Independent or semi-professional quantitative crypto trader running $50K–$2M in personal or friends-and-family capital.

**Context:**
Trades crypto perpetuals and spot on Binance, Bybit, or OKX. Has a Python-heavy workflow: Jupyter notebooks, pandas, a handful of custom scripts. Runs 5–20 strategy variants across 10–50 pairs. Iterates daily. Currently uses vectorbt free, a custom backtest loop, or ccxt + pandas. May have tried Backtrader and abandoned it because it's too slow for sub-minute data.

**Core problem:**
- Sweep iterations take hours when testing across multiple pairs and timeframes.
- Backtests ignore perpetual funding rates and realistic slippage, so live PnL consistently underperforms.
- Overfits parameters because they lack walk-forward tooling.

**Trigger moment:**
Deploys a strategy that looked great in backtest, loses money in first two weeks because funding costs and slippage ate the edge. Starts googling "realistic crypto backtester" or "vectorbt alternative faster."

**Budget:**
$500–3,000/year. This is a cost-of-doing-business tool. If it saves even one blown strategy deployment, it pays for itself in a day.

**Buying behavior:**
- Tries free tier / community edition first.
- Evaluates speed by running their existing strategy and comparing wall-clock time.
- Checks if perpetual funding and maker/taker fees are modeled.
- Reads Twitter/X threads and Discord reviews before buying.
- Will pay monthly ($50–150/mo) if there's a clear upgrade path.

**Objections:**
- "Can I replicate what I already do in vectorbt?"
- "Is the expression DSL flexible enough, or will I hit walls?"
- "What if the project dies in 6 months?"
- "I don't want to learn a new API."

**Distribution channels:**
- Crypto Twitter/X (CT), especially quant/algo subculture
- Discord communities (AlgoTrading, Quant, specific exchange servers)
- Telegram trading groups
- Reddit: r/algotrading, r/quant, r/CryptoCurrency
- YouTube quant channels (Part Time Larry, CodingTrading, etc.)

**Market value:** **HIGH PRIORITY** — First-mover segment. Crypto traders are comfortable with new tools, have urgency (24/7 markets), and the crypto-specific features (funding rates, borrow costs) are a unique differentiator that vectorbt lacks.

---

### ARCHETYPE 2: "The Micro-Team" — Small Crypto Trading Group

**User type:** Groupe informel de 2–5 traders/devs qui poolent du capital ($100K–$1M) et partagent la recherche. Pas un "fund" structuré — plutôt un groupe Telegram privé avec un repo Git partagé.

**Context:**
Typiquement des amis ou ex-collègues, souvent dev/data science de background, qui tradent crypto ensemble. Un ou deux codent les stratégies, les autres apportent du capital ou font du monitoring. Utilisent un mélange de scripts maison, vectorbt, et parfois freqtrade. Le "leader technique" fait 80% du travail de recherche dans des notebooks Jupyter partagés. Pas de structure légale formelle — c'est un Telegram group + un repo GitHub privé.

**Core problem:**
- Le gars technique utilise des scripts maison lents qui ne scalent pas quand le groupe veut tester plus de paires/timeframes.
- Pas de reproducibilité : quand il dit "Sharpe 1.8", personne ne peut vérifier indépendamment.
- Les autres membres du groupe ne peuvent pas facilement lancer des backtests eux-mêmes — trop technique.
- Quand une strat perd en live, c'est la guerre pour savoir si le backtest était foireux ou si le marché a changé.

**Trigger moment:**
Le groupe perd de l'argent sur une strat qui "marchait en backtest." Tensions internes. Le lead technique cherche un outil plus sérieux pour restaurer la confiance. Ou : le groupe veut scaler de 5 à 20 paires et les scripts maison n'y arrivent plus.

**Budget:**
$1,000–5,000/year partagé entre le groupe. Le lead technique décide seul de l'outil — les autres suivent. C'est un coût partagé qui reste modeste par tête ($200–1,000/personne).

**Buying behavior:**
- Le lead technique évalue seul, souvent en comparant la vitesse avec ses scripts existants.
- Veut un outil qu'il peut montrer aux autres : tearsheets propres, résultats reproductibles.
- Sensible au prix mais prêt à payer si la valeur est évidente (speed + réalisme).
- Pas de process d'achat formel — c'est une décision d'un soir sur Discord.

**Objections:**
- "On a déjà nos scripts, pourquoi changer ?"
- "Est-ce que les autres du groupe pourront l'utiliser sans être dev ?"
- "On n'a pas besoin d'un truc enterprise, juste quelque chose qui marche."

**Distribution channels:**
- Discord et Telegram (groupes de trading privés)
- Twitter/X crypto quant
- Bouche à oreille entre groupes similaires
- Reddit r/algotrading
- Le lead technique est souvent aussi un "Crypto Quant" (Archetype 1) qui a grandi

**Market value:** **MEDIUM-HIGH PRIORITY** — Revenue supérieur à l'individuel car le coût est partagé (plus facile à justifier). Le lead technique est le vrai décideur et se trouve dans les mêmes channels que l'Archetype 1. Convertir un Crypto Quant solo en lead technique d'un micro-team multiplie la valeur du client par 3–5x.

---

### ARCHETYPE 3: "The Signal Seller" — Strategy-as-a-Service Operator

**User type:** Runs a trading signal service, copy-trading platform, or managed account service. Sells access to strategies or signals to subscribers.

**Context:**
Operates 3–20 strategies across crypto and sometimes equities. Publishes track records to attract subscribers ($50–500/mo each, 50–5,000 subscribers). Needs to rapidly prototype new strategies, prove they work out-of-sample, and publish credible performance metrics. Currently uses a mix of TradingView Pine Script for signals and a custom Python backtest for track record generation.

**Core problem:**
- Needs fast parameter sweeps to find strategies that look good on a tear sheet.
- Must demonstrate robustness (walk-forward, out-of-sample) to retain subscribers.
- Subscribers churn if strategies underperform backtest claims → needs realistic execution modeling.
- Time-to-market matters: first to publish a strategy on a new narrative (e.g., "AI token momentum") wins subscribers.

**Trigger moment:**
Subscriber churn spikes because published track record diverged from live results. Or: competitor publishes walk-forward validated results and their own track record looks amateur by comparison.

**Budget:**
$1,000–5,000/year. The tool is a revenue multiplier: better strategies → lower churn → higher subscriber revenue.

**Buying behavior:**
- Evaluates based on quality of tear sheet output and speed of iteration.
- Cares deeply about visualization (plots, monthly return heatmaps) because these are customer-facing.
- Wants export to PDF / HTML for marketing.
- Will pay for white-labeling or embedding.

**Objections:**
- "Can I brand the output as my own?"
- "Does it generate the charts I need for my landing page?"
- "I need to backtest equities too, not just crypto."

**Distribution channels:**
- Twitter/X (fintwit, crypto Twitter)
- Collective2, Darwinex, eToro (copy-trading platforms)
- Trading communities (EliteTrader, Trade2Win)
- YouTube / content marketing

**Market value:** **MEDIUM-HIGH PRIORITY** — Good revenue potential, relatively easy to reach, and they become evangelists (their subscribers see the tearsheets and ask what tool generated them).

---

### ARCHETYPE 4: "The Quant Student" — Aspiring Quantitative Trader

**User type:** Graduate student in quantitative finance, financial engineering, or CS. Or a self-taught developer going through QuantConnect / Udemy / YouTube courses learning algo trading.

**Context:**
Learning Python, building their first strategies. Currently using free tools: Backtrader, basic pandas loops, or QuantConnect's free tier. Building a portfolio of strategy backtests for job applications or to start trading with small personal capital ($1K–$20K).

**Core problem:**
- Free tools are slow and teach bad habits (no slippage, no fees, no walk-forward).
- Wants to impress in quant interviews with realistic, reproducible research.
- Needs to learn fast: good docs, examples, and a shallow learning curve matter enormously.

**Trigger moment:**
Professor or mentor says "your backtest is unrealistic, you're not accounting for transaction costs." Or: rejected from a quant role because their take-home project used a naive backtester.

**Budget:**
$0–200/year. Students are extremely price-sensitive. This is a funnel segment: they convert into Archetype 1 or 2 within 1–3 years.

**Buying behavior:**
- Free tier or educational discount is mandatory.
- Evaluates based on documentation, tutorials, and community.
- Follows influencers who recommend tools.
- GitHub stars and community size matter as social proof.

**Objections:**
- "Backtrader is free, why would I pay?"
- "I don't have enough capital to justify a paid tool."
- "The API looks different from what I learned in class."

**Distribution channels:**
- Reddit: r/algotrading, r/quant, r/QuantFinance
- YouTube educational channels
- University partnerships / academic licenses
- Kaggle, QuantConnect community
- Dev.to, Medium quant articles

**Market value:** **MEDIUM PRIORITY** — Low direct revenue but critical for ecosystem growth, community building, and pipeline into higher-value segments. Free tier + educational pricing.

---

### ARCHETYPE 5: "The Platform Builder" — Petite Fintech / Outil de Trading

**User type:** Dev lead ou CTO d'une petite startup (2–10 personnes) qui construit un produit autour du trading : bot-as-a-service, plateforme de copy-trading, outil de portfolio analytics, ou dashboard pour traders.

**Context:**
Startup early-stage, souvent bootstrapped ou avec un petit seed. Le fondateur/CTO est technique et code lui-même. Ils ont besoin d'un moteur de backtest intégré dans leur produit mais n'ont ni le temps ni les ressources pour en construire un from scratch. Actuellement ils utilisent un hack maison (pandas + cron jobs) qui tient avec du scotch.

**Core problem:**
- Construire un backtester production-grade prendrait 3–6 mois de dev qu'ils n'ont pas.
- Le hack maison craque dès que les utilisateurs augmentent.
- Besoin d'un moteur embeddable via API, pas d'un outil standalone.

**Trigger moment:**
Premier vrai utilisateur payant se plaint que le backtest est lent/buggé. Ou : le fondateur réalise qu'il passe 50% de son temps à maintenir le backtester au lieu de construire son produit.

**Budget:**
$5,000–20,000/year. C'est un coût infra, comme Stripe ou Supabase. Justifiable si ça remplace 3 mois de dev interne.

**Buying behavior:**
- Le CTO évalue seul, en quelques jours.
- Compare le coût de licence vs. le temps de dev économisé.
- Veut une API propre, de la doc, et des exemples d'intégration.
- Pas de process formel — décision rapide si le produit fait le job.

**Objections:**
- "Et si le projet meurt ? On est bloqués."
- "On a besoin de custom : equities, forex, pas que crypto."
- "Le pricing scale avec nos utilisateurs ? Ça peut devenir cher."

**Distribution channels:**
- GitHub (les CTOs cherchent des libs open-source à intégrer)
- Indie Hackers, Hacker News
- Twitter/X (startup/dev community)
- Product Hunt (quand le produit est prêt)

**Market value:** **MEDIUM, LONG TERME** — Revenue correct par client mais nécessite que le produit ait une API d'intégration stable et une licence commerciale claire. À poursuivre en Year 2 quand le core product a fait ses preuves.

---

### ARCHETYPE 6: "The Newsletter Trader" — Content-First Trader

**User type:** Runs a paid trading newsletter (Substack, Beehiiv) or a premium Discord/Telegram group. Trades part-time. Revenue comes from content, not from trading PnL.

**Context:**
Publishes weekly or daily market analysis with specific trade ideas. Wants to backtest ideas before publishing to maintain credibility. Currently uses TradingView for charting and a rough spreadsheet or pandas script for backtesting. Has 500–10,000 subscribers paying $10–$50/mo.

**Core problem:**
- Needs quick validation: "does this idea actually work historically?"
- Wants to publish credible tear sheets alongside trade ideas.
- Can't spend hours on each backtest — needs fast iteration.

**Trigger moment:**
A subscriber calls them out: "Your last 5 trade ideas lost money, did you even backtest these?" Or: a competing newsletter starts publishing walk-forward validated results.

**Budget:**
$200–1,000/year. The tool pays for itself if it prevents one bad trade idea from reaching subscribers (churn prevention).

**Buying behavior:**
- Values simplicity over power. Doesn't want to write Python — wants the simplest possible interface.
- Impressed by beautiful charts and exportable tear sheets.
- Will pay for a "publish" button that generates a shareable report.

**Objections:**
- "I'm not a developer, is this too technical?"
- "TradingView does backtesting, why do I need this?"
- "I just need basic stuff — moving averages, RSI."

**Distribution channels:**
- Substack / Beehiiv communities
- Twitter/X fintwit
- Trading Discords
- YouTube finance creators

**Market value:** **LOW-MEDIUM PRIORITY** — Moderate revenue, but these users need a simpler UI layer (web app, not Python SDK). Best served later when the hosted product exists.

---

## 4. Budget Analysis

| Archetype | Annual Budget | Pricing Model | Price Point | Justification |
|-----------|---------------|---------------|-------------|---------------|
| **Crypto Quant** | $500–3,000 | Monthly subscription | $49–149/mo | Predictable cost, low commitment, scales with usage |
| **Micro-Team** | $1,000–5,000 | Monthly sub (team plan) | $99–249/mo | Partagé entre 2–5 personnes, multi-seat |
| **Signal Seller** | $1,000–5,000 | Monthly subscription | $99–249/mo | Higher tier with white-label charts, export features |
| **Quant Student** | $0–200 | Freemium + edu discount | Free / $9–19/mo | Free tier converts to paid in 1–3 years |
| **Platform Builder** | $5,000–20,000 | Annual license | $5K–20K/yr | Coût infra, remplace du dev time |
| **Newsletter Trader** | $200–1,000 | Monthly subscription | $29–79/mo | Simple tier with chart exports |

### Pricing Model Rationale

**Subscription (monthly/annual) for individuals:**
- Quants iterate continuously — they need ongoing access, not a one-time tool.
- Monthly reduces friction to start; annual discount incentivizes commitment.
- Subscription aligns revenue with ongoing value delivery (new indicators, connectors, features).

**Team plan pour micro-teams:**
- Même produit que l'individuel mais multi-seat (2–5 users).
- Prix légèrement réduit par tête pour encourager l'adoption groupe.
- Pas besoin de SLA ou support enterprise — ce sont des devs, ils se débrouillent.

**Do NOT use performance fees:**
- Difficult to verify and enforce.
- Creates misaligned incentives (users avoid reporting wins).
- Adds legal complexity.
- Quant tools historically fail with performance-fee models.

**Do NOT use one-time licenses:**
- Kills recurring revenue.
- No incentive to maintain and improve the product.
- Users expect ongoing updates in a fast-moving market.

---

## 5. Which Segment to Target First

### Primary: The Crypto Quant (Archetype 1)

**Why this segment wins on every dimension:**

1. **Highest urgency:** 24/7 crypto markets mean they feel pain continuously. Every day with a bad backtester is a day of lost edge.

2. **Unique product-market fit:** Perpetual funding rates, borrow costs, and maker/taker fee modeling are crypto-specific features that no competitor handles well. This is a defensible wedge.

3. **Easiest to reach:** Crypto quants congregate in known, accessible communities (Twitter/X, Discord, Telegram, Reddit). A single viral thread can generate thousands of signups.

4. **Fast evaluation cycle:** They can `pip install`, run their strategy, and see the speed difference in 10 minutes. No procurement process, no committee.

5. **Willingness to pay:** They are spending real capital. A $50/mo tool that prevents one bad trade is an obvious purchase.

6. **Network effects:** Crypto quants talk to each other. One happy user in a Discord generates 5 more.

7. **Pipeline to higher segments:** Today's crypto quant is tomorrow's fund PM. Lock them in early.

### Secondary: Les Micro-Teams (Archetype 2)

Pas besoin de les "closer" séparément — ce sont des Crypto Quants (Archetype 1) qui ont grandi. Convertir un utilisateur solo satisfait en lead technique d'un groupe est le chemin naturel. Proposer un team plan quand un utilisateur demande "est-ce que mon pote peut avoir un compte aussi ?"

### Sequencing

```
Month 1–6:   Crypto Quants (community, content, free tier → paid conversion)
Month 3–9:   Signal Sellers (adjacent aux crypto quants, mêmes channels)
Month 6–12:  Micro-Teams (upsell naturel des Crypto Quants existants → team plan)
Month 12–24: Platform Builders (enterprise, seulement quand le produit est mature)
```

---

## 6. Which Segments to Ignore (For Now)

### Ignore: Retail Hobby Traders (the majority of r/algotrading)

**Why:** They have no budget, no urgency, and no willingness to pay. They want free tools, will complain about any price point, and generate disproportionate support burden. The free tier will serve them, but do not optimize for this segment.

### Deprioritize: Newsletter Traders (Archetype 6)

**Why:** They need a web UI, not a Python SDK. Serving them requires building an entirely different product surface (hosted web app with a visual strategy builder). This is a valid segment for Year 2+ when the hosted product exists, but building for them now would distract from the core SDK product.

### Deprioritize: Quant Students (Archetype 4)

**Why:** Important for long-term ecosystem growth but generate near-zero revenue. Serve them with a generous free tier and good documentation. Do not build features specifically for this segment. They will self-serve if the product is good.

### Ignore: Traditional Finance / Equities-Only Traders

**Why:** The product is currently crypto-optimized (Binance connector, funding rates, 1-second bars). Equities traders need different data sources (polygon.io, IEX), different fee structures (SEC fees, ECN rebates), and different market microstructure (exchanges, dark pools). Expanding to equities is a valid roadmap item but not a launch priority.

### Ignore: Hedge Funds, Prop Desks, HFT

**Why:** Les vrais fonds quant (Renaissance, Two Sigma, Jump, etc.) et même les petits fonds structurés ($10M+) construisent TOUT en interne. C'est leur avantage compétitif. Leur infra de backtest est propriétaire, maintenue par des équipes dédiées, et intégrée à leur stack d'exécution. Ils n'achèteront jamais un outil externe pour ça — au mieux ils embauchent un dev de plus. Les prop desks sérieux (Jane Street, Optiver, DRW) c'est pareil en pire : tout est custom C++/OCaml/FPGA. Ne pas perdre de temps à cibler ce segment — c'est un marché fermé.

---

*Generated 2026-03-14 for backtester-engine go-to-market planning.*
