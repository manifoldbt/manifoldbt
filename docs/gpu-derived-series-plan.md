# Plan — Séries dérivées hoistées sur GPU (« option C ») (2026-07-17)

Objectif : permettre aux ops fenêtrées du transpileur (`ZScore`, `RollingMean`,
`Rsi`, `Roc`, futurs `RollingStd/Sum/...`) de prendre une **expression** en
entrée, pas seulement une colonne brute — ce qui débloque l'exemple 16
(`zscore(price_ratio - hr_ratio, zwin)`) et toute la famille « indicateur d'un
spread/ratio » sur GPU.

## Le problème (analysé le 2026-07-17, voir la discussion du rapport)

Le kernel évalue en **streaming** : les intermédiaires vivent une barre dans des
registres. Une op fenêtrée doit évincer la valeur qui sort de la fenêtre
(`sum -= x[jj-w]`) : pour une colonne (`CCOL`) ou une série hoistée (`HIND`)
c'est un accès mémoire ; pour une expression dérivée, la valeur d'il y a `w`
barres a été jetée. Le CPU n'a pas ce problème : son évaluateur matérialise
chaque nœud du DAG en array.

Seul `EwmMean` accepte une expression aujourd'hui — une EMA ne regarde jamais
en arrière. `RollingMean`, `Rsi`, `ZScore` exigent tous une colonne brute ;
cette limite est **préexistante et structurelle**, pas un choix de mon ZScore.

## Le design

**Matérialiser l'expression d'entrée comme lignes `hind` supplémentaires**,
dédupliquées par résolution de paramètres, remplies par un kernel généré ; l'op
fenêtrée reste un **fold in-thread** (par combo) qui lit la ligne dérivée en
accès aléatoire (`HINDAT(j, jj)` / `HINDAT(j, jj-w)`).

```
stage 1  hoist_fill    : indicateurs feuilles (SMA/EMA/RSI/ROC/ZScore sur colonnes)
stage 2  derived_fill  : expressions POINTWISE sur (cols ∪ lignes stage 1)   ← NOUVEAU
sim      sweep_t       : folds fenêtrés in-thread lisant les lignes dérivées ← ÉTENDU
```

Décisions clés :

- **Éligibilité** : l'entrée doit être pointwise sur (colonnes brutes ∪ feuilles
  hoistables ∪ littéraux ∪ params). Sinon → CPU, raison nommée (Phase 0 style) :
  « windowed op over a non-pointwise expression ».
- **Dédup** : clé = empreinte structurelle de l'expression + bits résolus de
  CHAQUE param référencé (vecteur, pas un seul u64 — une expression peut lire
  plusieurs params). Même mécanique HashMap que `build_hoist_plan`. Nombre de
  lignes = résolutions distinctes de l'ENTRÉE (ex. 16 : #smooth distincts, PAS
  #(smooth × zwin) — la fenêtre reste par combo, in-thread). C'est ce qui rend
  la VRAM raisonnable à 1M combos.
- **`derived_fill` est un kernel GÉNÉRÉ** (comme `sweep_t`), pas un kernel fixe :
  le corps pointwise vient du MÊME `PipeCtx::emit` que le sim (mêmes sémantiques
  NaN, division-par-zéro → NaN, etc. — zéro seconde implémentation qui dérive).
  Pointwise ⇒ barres indépendantes ⇒ un thread par (ligne, barre), coalescé,
  ~1 ms pour 1000 lignes × 44k barres. Lancé entre hoist_fill et sweep_t.
- **Chaînage limité en V1** : dérivé = pointwise uniquement. Un fenêtré DANS un
  fenêtré sur expression (`zscore(sma(expr))`) → CPU, raison nommée. (Extensible
  plus tard en étages topologiques ; hors périmètre V1.)
- **Folds in-thread** : sémantique `*_no_nulls` de bt-expr AU COMPLET, y compris
  `nan_count` — voir P0 ci-dessous. État ZScore in-thread ≈ 4-6 registres.
- **fp32 scan** : lignes dérivées calculées en f64, stockées f32 en mode scan
  (même argument que le float-hind de la Part 5 : identique au cast côté
  lecteur).
- **Garde VRAM** : bytes(hind + derived) > free/2 → CPU, raison nommée.
- **Multi-asset** : hors V1 (comme l'exo), raison nommée.

## Bit-identité (la chaîne de preuve)

CPU : matérialise l'entrée (array) puis `zscore_no_nulls(vals)`. GPU : la ligne
dérivée doit être **bit-identique à l'array CPU** (mêmes ops pointwise dans le
même ordre via le même emit(), fmad off, mêmes sources : cols identiques,
lignes stage-1 déjà pinnées à bt-expr) ; puis le fold in-thread refait
exactement `zscore_no_nulls` sur les mêmes valeurs. Chaque maillon testé.

## Étapes

### P0 — Bug latent découvert par l'analyse (à faire AVANT tout)
Le fold SMA **in-thread** (`ss += CCOL(...)`, chemin non-hoisté = multi-asset)
n'a AUCUNE gestion NaN, alors que `rolling_mean_pair` (bt-expr) exclut les NaN
de ses sommes (`bad_a`/`bad_b`). Divergence silencieuse sur données à trous en
multi-asset — même famille que `804de5e`, invisible parce que les tests NaN
tournent en mono (hoisting actif) et que le e2e multi n'a pas de NaN.
- Auditer les 3 folds in-thread (SMA / RSI / ROC) contre bt-expr sur NaN.
- Test : e2e multi avec NaN closes (étendre `gpu_sweep_multi_e2e`), vérifier
  qu'il MORD (rituel du bug délibéré), corriger les folds.
- Estimation : ½ journée. Indépendant de C mais bloquant : les folds dérivés
  réutiliseront ce code.

### P1 — Enregistrement des lignes dérivées (host)
- `PipeCtx` : détection d'éligibilité (fn `is_pointwise_over_leaves`),
  empreinte, `derived_slot(expr) -> dj`, table `derived: Vec<DerivedSpec>`.
- `build_hoist_plan` étendu : résolution par combo des params de chaque
  expression dérivée, dédup, `hrow` étendu aux lignes dérivées, garde VRAM.

### P2 — Kernel `derived_fill` généré + wiring
- Génération du source (corps par ligne via emit(), switch sur le kind de
  ligne), compile NVRTC (cache par hash), launch stage 2, slot dans
  `MBT_GPU_PROBE`.
- `ptxas -v` sur le kernel généré (attendu trivial : pointwise, pas d'état).

### P3 — Folds in-thread sur lignes dérivées
- Macro `HINDAT(j, idx)` (HIND à index arbitraire).
- `ZScore` d'abord (débloque ex. 16), puis `RollingMean`, `Rsi`, `Roc` :
  entrée Column → chemins existants ; entrée dérivée-éligible → fold in-thread
  avec sémantique bt-expr complète.
- **GATE ptxas** : le sim est à 48/48 registres. Si le fold in-thread fait
  spiller, relâcher `__launch_bounds__` UNIQUEMENT pour les sources générées qui
  contiennent des folds in-thread (le bound est par source générée, donc par
  sweep) — mesurer, pas supposer.

### P4 — Tests (chaque test vérifié MORDANT via bug délibéré)
- `derived_fill_matches_bt_expr` : ligne dérivée vs array intermédiaire bt-expr,
  barre par barre, NaN gaps inclus (harnais `hoist_fill_matches_bt_expr` étendu).
- Fold in-thread vs bt-expr : zscore(expr) complet, barre par barre.
- e2e forme exemple 16 (exo + spread + zscore, périodes en params) :
  bit-identité CPU==GPU + non-vacuité + appel direct `run_sweep_lite_gpu`
  (une retombée = échec, pas une comparaison CPU-vs-CPU silencieuse).
- Retombées nommées : non-pointwise, fenêtré imbriqué, VRAM, multi-asset.

### P5 — Bench + rapport
- Exemple 16 réel (hashrate), grille ~10k : cible GPU >> CPU (aujourd'hui :
  2 589 c/s CPU, GPU retombe). Section dans `gpu_funding_bench.md`.
- Mettre à jour `docs/gpu-sweep-coverage-plan.md` (la Phase 2b absorbe C).

## Estimation & risques

| | |
|---|---|
| P0 | ½ j (correction comprise si le bug se confirme) |
| P1+P2 | ~1 j |
| P3 | ½ j + gate registres |
| P4+P5 | ½ j |
| **Total** | **~2-2,5 jours** |

Risques : (1) pression registres des folds in-thread — gate ptxas, bounds par
source ; (2) explosion du nombre de lignes dérivées si une expression référence
plusieurs params à forte cardinalité — garde VRAM + retombée nommée ; (3) toute
divergence NaN — couverte par les tests barre-par-barre ancrés bt-expr, jamais
par des miroirs.

---

## STATUT : LIVRÉ (2026-07-17)

- P0 : fold SMA in-thread corrigé (bad-count miroir de `rolling`), RSI/ROC
  audités conformes ; e2e `gpu_multi_sma_nan_volume_{2,5}` vérifié mordant.
- P1-P2 : `DerivedReg`/`derived_slot`, plan étendu (dédup multi-params, hrow
  absolu, garde VRAM sur le total), `derived_fill` généré par le même emit
  (macros remappées), `hoist_fill` a un stride explicite.
- P3 : fold ZScore in-thread sur ligne dérivée via `HINDAT` ; GATE ptxas :
  bounds relâchés à 7 (f64) / 8 (fp32) blocs pour les sources à folds, zéro
  spill ; fold-free inchangé (48 regs @10).
- P4 : `derived_fill_matches_bt_expr` (mord sur +1e-12 que l'e2e ne voit pas),
  e2e bit-identité + retombée nommée fenêtré-imbriqué.
- P5 : exemple 16 réel, 102x à 30k combos, Part 11 du rapport.
- Reste (hors V1, repris en Phase 2b) : `RollingMean`/`Rsi`/`Roc` sur entrée
  dérivée (mécanique identique au ZScore), multi-asset, chaînage fenêtré.
