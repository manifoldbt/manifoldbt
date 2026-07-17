# Plan — Couverture GPU du sweep (2026-07-17)

Contexte : le sweep GPU fait 300k combos/s (fp64 bit-identique) / 676k (fp32 scan),
mais seules les stratégies dans un périmètre étroit en profitent. Tout le reste
retombe sur le CPU (~1-20k combos/s) **en silence**. Ce plan séquence l'élargissement
du périmètre, du moins cher au plus cher, avec un gate data-driven au milieu.

État de départ : PR #74 (18 commits GPU) en attente de merge ; branche
`perf/python-metrics-extraction` (3 commits : dicts directs, `sweep_columns`,
fix du pool VRAM) prête. Les phases ci-dessous partent au-dessus.

Log de mesures : `gpu_funding_bench.md`. Toute nouvelle phase y ajoute sa section.

---

## Les trois couches qui bloquent une stratégie

1. **Ops transpileur manquants** : `Eq, Lag, Lead, RollingStd, RollingSum,
   RollingMin, RollingMax, Scan` (→ Kalman et tout indicateur à état custom).
2. **Périmètre fast-path** (11 conditions) : la plus coûteuse est probablement
   `orders.is_empty()` — **toute stratégie avec SL/TP retombe**. Aussi :
   AtClose only, FixedBps only, frais mono-venue, pas de borrow rate, etc.
3. **Garde-fous sweep** : MTF, exogène, multi-source, cross-sectionnel,
   funding multi-asset.

Tri technique de la couche 3 : funding multi = ~15 lignes (terme déjà écrit pour
le mono) ; exogène = ~30 lignes (l'exo est ASOF-joint sur la grille de barres au
chargement → colonne ordinaire pour le kernel) ; cross-sectionnel = moyen (le
kernel multi a déjà tous les symboles dans le thread, restructuration en 2
passes) ; MTF = moyen-lourd ; multi-source = lourd.

---

## Phase 0 — Rendre la retombée visible (≈1 h) — À FAIRE EN PREMIER

**Problème** : `run_sweep_lite(device="cuda")` retombe en silence. L'utilisateur
paie Pro, voit 12k combos/s, ne saura jamais que retirer son SL/TP le rendrait
25x plus rapide. Et nous, on priorise les phases suivantes à l'aveugle.

**Livrable**
- La raison de retombée (déjà construite : `gpu-sweep-unsupported: <raison>`)
  remonte à l'utilisateur : `warnings.warn` une fois par sweep, et champ
  `profile["gpu_fallback_reason"]` sur les résultats.
- Les raisons sont des chaînes stables (elles le sont déjà) → agrégeables plus
  tard en télémétrie si souhaité.

**Validation** : sweep avec SL/TP → warning avec la raison exacte ; sweep GPU
nominal → aucun warning, champ absent ; e2e inchangés.

**Gate de décision** : après quelques jours d'usage réel (et un tour des
exemples/notebooks), la fréquence des raisons décide l'ordre des phases 2-4.
Sans données, l'ordre ci-dessous est notre meilleure estimation.

---

## Phase 1 — Quick wins périmètre (≈½ journée)

### 1a. Funding multi-asset dans `sweep_t_multi` (~15 lignes)
Le terme funding est déjà dans `sweep_t` (mono) : `capital += -position * close
* rate` par barre, par symbole ici. Retirer le garde-fou
`"multi-asset funding not yet on the GPU kernel"`.
- **Validation** : colonne funding synthétique injectée, e2e multi bit-identique
  CPU==GPU (même harnais que le fix funding mono).

### 1b. Colonnes exogènes dans le sweep GPU (~30 lignes)
`load_exo_columns` ASOF-joint déjà l'exo sur `master_timestamps` → un
`Float64Array` de `num_bars`, indistinguable de `close` pour le kernel. Étendre
la résolution des `col_names` du transpileur aux colonnes exo alignées, retirer
`!config.exo_data.is_empty()` du garde-fou (garder `exo_sources`/multi-source
exclus).
- **Validation** : e2e avec exo synthétique bit-identique CPU==GPU ; l'exemple
  `16_hashrate_exogene.py` passe sur GPU ; NaN de tête d'ASOF (avant le premier
  point exo) couverts par le test.
- **Attention** : chemin multi-asset — un exo est global (pas par symbole), le
  layout `cols [n_cols][n_sym][num_bars]` duplique par symbole ou garde-fou
  multi+exo conservé en V1.

---

## Phase 2 — Ops transpileur faciles (≈½ journée) — **FAIT 2026-07-17**

**Livré** : ZScore/RollingStd/RollingSum/RollingMean/Rsi/Roc sur colonne OU
expression pointwise (séries dérivées), Eq, RollingMin/Max (hoistés, rescan
exact), scan sans param → colonne hôte. **Refusé avec raison nommée** :
Lag/Lead — mur sémantique null-vs-NaN (le CPU garde la position sur cible
null, le kernel n'a qu'un encodage) ; il faudrait un canal de validité.
Détails : Part 14 du rapport.

`ZScore`, `RollingStd`, `Eq`, `Lag`, `Lead`, `RollingSum`, `RollingMin`, `RollingMax`.

Débloque : Bollinger (rolling_std), breakouts (rolling_max/min), signaux à
retard (lag), et retire un pan de retombées silencieuses.

**`ZScore` en tête de liste** : mesuré 2026-07-17, c'est ce qui bloque
`examples/16_hashrate_exogene.py` (la Phase 1b a débloqué son exo, mais il bute
maintenant sur `indicator ZScore is not supported`). ZScore est une op à part
entière dans bt-expr, pas un `rolling_std` déguisé — je l'avais manquée dans
l'inventaire initial ; c'est le warning de la Phase 0 qui l'a nommée.

**Mise à jour 2026-07-17 (suite)** : ZScore-sur-colonne est livré (`d883edd`),
mais l'exemple 16 fait `zscore(EXPRESSION)` et retombe encore : TOUTES les ops
fenêtrées du transpileur exigent une colonne brute (limite structurelle
préexistante — l'éviction `sum -= x[jj-w]` exige un accès aléatoire que le
streaming n'a pas ; seule l'EMA, sans regard arrière, accepte une expression).
Le correctif structurel est le **plan « séries dérivées »** :
[`gpu-derived-series-plan.md`](gpu-derived-series-plan.md) (~2-2,5 j, P0 = bug
NaN latent des folds in-thread multi-asset découvert par l'analyse).

**Mise à jour 2026-07-17 (fin de journée) : plan séries dérivées LIVRÉ
(P0-P5).** Le bug NaN du fold SMA in-thread est corrigé (e2e multi NaN-volume,
rouge-puis-vert), et `zscore(EXPRESSION)` tourne sur GPU : l'entrée pointwise
est matérialisée en lignes dérivées (kernel `derived_fill` généré par le même
emit), le fold z-score lit la ligne via `HINDAT` in-thread. Exemple 16 réel :
**102x** à 30k combos (0.43 s vs 43.6 s CPU), bit-identique. Rapport : Part 11
de `gpu_funding_bench.md`. Reste de la Phase 2 : brancher `RollingMean`/`Rsi`/
`Roc` sur le même chemin dérivé, puis les ops manquantes ci-dessus.

**Trou distinct découvert au passage** : une stratégie SANS périodes dynamiques
part dans `transpile_sizing` (pointwise uniquement) et meurt sur le premier
`EwmMean`, bien avant le pipeline. L'exemple 16 tel quel tombe là-dessus. Peu
grave pour un sweep (on paramètre les fenêtres, donc on passe par
`transpile_pipeline`), mais à garder en tête pour `run_batch`.

**Règles héritées de la campagne (non négociables)**
- Chaque fold miroir **bt-expr op-pour-op**, y compris NaN : le bug des folds
  NaN (Part 5) venait exactement d'un miroir non ancré au moteur.
  Pièges connus : `Eq` = `(a-b).abs() < f64::EPSILON` (pas `==`) ;
  `RollingStd` : même formule de Welford/somme que bt-expr, même fenêtre sale.
- Tests de parité **contre bt-expr** (pattern `gpu_fold_nan_gaps_*`), jamais
  contre un miroir du même algorithme.
- Hoisting : chaque nouveau kind hoistable rejoint `hoist_fill` (kind + period),
  sinon état par thread via `state_decls` — attention au **cap 48 registres**
  (ptxas -v systématique, la table des résultats négatifs Parts 4-5 fait foi).

---

## Phase 3 — `scan` sur GPU → Kalman — **FAIT 2026-07-17**

**Livré** : gate 3a passé (bande Kalman dépliée + fold = 94 regs, zéro
spill) ; `lower_scan_tape` extrait dans bt-expr (une bande, deux exécuteurs) ;
traduction 1:1 ScanOp→CUDA (Eq EPSILON du VM répliqué, ≠ du == array) ;
`derived_fill` séquentiel par ligne → fenêtré-sur-scan-sweepé OK (dédup par
bits résolus) ; fix param()-dans-scan (has_dynamic_periods + collecteur
Python). Surface (q,w) 1M réelle : 25.8 s. Refus nommés : log/exp, inits
non-pointwise, colonnes à nulls. Phase 5 (dispatch auto) : device="auto",
seuil mesuré 1k combos, Part 15.

## Phase 3 (plan d'origine, conservé pour référence)

### 3a. Sonde registres (30 min, GATE)
Écrire À LA MAIN le kernel Kalman (bande ScanOp dépliée en CUDA), `ptxas -v`
sm_86. Le kernel sim est à 48/48 registres **zéro marge** : si la bande scan
spille, le gain s'évapore → **stop, on documente, on ne construit pas le
générateur**.

### 3b. Générateur ScanOp → CUDA
La bande `CompiledScan.instructions` est en SSA plat (24 opcodes scalaires) →
traduction mécanique, 1 ligne CUDA par opcode, branchée sur
`state_decls`/`state_init`/`body` existants.
Sémantiques à mirrorer exactement (source de vérité : `evaluator.rs`) :
- `Div` : dénominateur `== 0.0` → **NaN** (pas l'inf IEEE)
- `Eq` : `(a-b).abs() < f64::EPSILON`
- init : `prev_state[i] = init_arr.value(0)`, null → `0.0`
- ordre d'exécution : registre = pointeur d'instruction, writeback après output

### 3c. `param()` dans un nœud scan (bug indépendant, petit)
`CompiledScan.param_names` existe déjà ; c'est la **découverte des paramètres de
stratégie** qui ne descend pas dans les nœuds scan ("uses undefined parameters:
q"). Fix côté validation/collecte → débloque le sweep de `q` (la dimension la
plus intéressante du filtre), même sur CPU.

### 3d. Acceptance
- Parité GPU vs VM bt-expr bit-identique, NaN gaps inclus.
- e2e sweep Kalman CPU==GPU bit-identique.
- **La heatmap 1M Kalman (w×en, puis q×en) sur GPU** — l'objectif d'origine.
  Budget attendu : quelques secondes au lieu de 15 min CPU.
- fp32 : vérifier le ranking (le scan ajoute des Div/Sqrt → sensibilité à
  mesurer, pas à supposer).

---

## Phase 4 — SL/TP sur le fast path GPU (gros ; décision APRÈS Phase 0)

`orders.is_empty()` est probablement le garde-fou le plus coûteux en usage réel
(hypothèse à confirmer par les données de la Phase 0). Résolution intra-bar
SL/TP = high/low + sémantique de priorité déjà fixée à la 0.12.0 côté lite CPU.
Design séparé — ne pas commencer avant d'avoir les données de fréquence et le
merge des phases précédentes.

## Phase 5 — Dispatch GPU/CPU (à faire À LA FIN)

**Mesuré 2026-07-17** : sur une grille de 100 combos, le GPU est **6x plus LENT**
que le CPU (882 vs 5 327 combos/s) — les coûts fixes (upload, launch, hoist,
compile) écrasent le gain. À 6 000 combos le GPU gagne 7x. Il existe donc un
seuil de rentabilité, et aujourd'hui l'utilisateur qui passe `device="cuda"` sur
une petite grille **paie pour être ralenti, sans rien qui le prévienne**.

Deux options, à trancher le moment venu :
1. **Bascule automatique** sous un seuil (attention : `device="cuda"` explicite
   qui exécute sur CPU, c'est exactement le genre de magie silencieuse que la
   Phase 0 vient de supprimer — il faudrait le dire).
2. **Warning symétrique** : « grille trop petite pour amortir le GPU (N combos,
   seuil ~M) », et laisser l'utilisateur décider.

Le seuil dépend de num_bars, n_sym et de la complexité du pipeline : à calibrer
par mesure, pas à coder en dur (cf. [[feedback_portability_first]] : pas de
seuil spécifique à une machine). Placé en fin de plan volontairement : c'est du
polish, les phases 1-3 débloquent des stratégies entières.

## Phase 6 — Backlog (ordre selon Phase 0)

- Cross-sectionnel dans `sweep_t_multi` (2 ops, restructuration 2 passes).
- Exo en multi-asset (broadcast de la série globale dans chaque slab symbole).
- MTF (grilles multiples, mapping d'index par timeframe).
- Multi-source / cross-exchange (chemin général, lourd).
- Frais par venue, borrow rate, AtOpen (extensions du fast path).
- `transpile_sizing` (chemin sans params) limité au pointwise → bloque `run_batch`
  sur toute stratégie à indicateurs.

---

## Invariants transverses (toutes phases)

1. **Bit-identité fp64 CPU==GPU** : chaque phase ajoute ses cas au harnais e2e
   (`to_bits` par métrique). Les checksums 21-métriques du bench servent de
   non-régression (`14fabff006fda0a0` fp64 / `c1be6db787e2f49f` fp32 sur la
   grille de référence).
2. **Parité ancrée à bt-expr**, jamais à un miroir (leçon Part 5).
3. **ptxas -v à chaque changement de kernel** (cap 48 regs, spills interdits).
4. **Un garde-fou ne se retire qu'avec le test qui prouve le remplacement.**
5. **Chaque phase = sa branche + sa PR + sa section dans le bench log.**
   Livraison incrémentale, pas de méga-branche de 18 commits une deuxième fois.

## Estimation d'ensemble

| Phase | Effort | Gain |
|---|---|---|
| 0 — retombée visible | ~1 h | priorisation data-driven + UX Pro |
| 1 — funding multi + exo | ~½ j | 2 familles débloquées, quasi sans risque |
| 2 — 7 ops faciles | ~½ j | Bollinger/breakout/lag sur GPU |
| 3 — scan → Kalman | 1-2 j (gate 30 min) | la stratégie de l'article sur GPU + échappatoire universel |
| 4 — SL/TP | gros | à chiffrer après Phase 0 |
