# Critical Analysis

> **Data note.** Metrics in this analysis are from the prototype run `run_20260210_153030`,
> which used 385,440 synthetic observations over a simulated 365-day period. The synthetic
> data was designed to mimic real TfL delay patterns (line baselines, peak crowding, weather
> effects, rare spikes). A later run on 44 real TfL observations (`run_20260310_164049`)
> produced much worse results (LightGBM MAE 12.4), showing how important data volume is.
> The analysis scripts in `analysis/` work on the test split (748 rows, final 20% of one
> simulated week).

## 1. Why the Model Works — Feature Importance

The LightGBM model gets a test MAE of 2.01 minutes, well below the naive baseline (3.62) and
slightly below Ridge regression (2.03). The ablation study (`analysis/ablation_study.py`) tests
each feature group's contribution by training LightGBM with each group removed in turn and
comparing against the full 37-feature baseline MAE of 4.638.

The results are pretty surprising:

| Removed Group | MAE | Δ vs Full Model | % Change |
|---|---|---|---|
| *(naive mean baseline)* | 5.729 | +1.091 | +23.5% |
| *(full model — all features)* | 4.638 | 0.000 | 0.0% |
| historical | 4.982 | **+0.344** | **+7.4%** |
| temporal | 4.676 | +0.038 | +0.8% |
| network | 4.334 | −0.304 | −6.5% |
| weather | 4.304 | −0.334 | −7.2% |
| line_metadata | 4.187 | −0.451 | **−9.7%** |

Only **historical features** (lags, rolling means) actually help — removing them raises MAE by
0.344 (+7.4%). This makes sense: tube delays are strongly autocorrelated. When a service falls
behind, fixed headways propagate the disruption forward, and `lag_delay_1` captures this directly.

More interesting is that removing the **network**, **weather**, and **line_metadata** groups each
*reduces* MAE. Removing line_metadata improves MAE by 9.7%; weather by 7.2%; network by 6.5%.
These features are adding noise, not signal. With only 598 training rows and 37 features, the
model is overfitting on these groups. The learning curves (§2) confirm this: train MAE is 1.73
vs val MAE of 4.64 — a 2.91 minute gap, classic overfitting. The temporal group is basically
neutral (+0.8%).

So the model's good headline performance (MAE 2.01) is really down to RandomizedSearchCV finding
good regularisation parameters (`num_leaves=15`, `subsample=0.7`, `min_child_samples=20`) that
suppress the noise from the extra features. The ablation (fixed hyperparameters, no search)
exposes the underlying issue: at 598 rows, the lag structure alone is doing most of the work.

The **SHAP analysis** from `explain.py` backs this up — lag and rolling features dominate global
importance, while weather and network features contribute much less.

---

## 2. Where the Model Fails — Per-Line Error Breakdown

The per-line analysis (`analysis/per_line_performance.py`) shows big differences in accuracy
across lines:

| Line | MAE | RMSE | N |
|---|---|---|---|
| Waterloo & City | 3.308 | 3.450 | 14 |
| Jubilee | 3.953 | 4.096 | 14 |
| Bakerloo | 4.125 | 4.226 | 14 |
| Central | 4.137 | 4.257 | 13 |
| Hammersmith & City | 4.180 | 4.254 | 14 |
| Northern | 4.226 | 4.402 | 14 |
| District | 4.262 | 4.383 | 13 |
| Circle | 4.595 | 4.677 | 14 |
| Victoria | 9.597 | 14.994 | 13 |
| Metropolitan | 15.214 | 17.632 | 13 |
| Piccadilly | 15.753 | 17.800 | 14 |

With only 13–14 test samples per line, R² is basically meaningless (ranges from −0.6 to −140),
so this analysis focuses only on MAE and RMSE here.

**Waterloo & City** has the lowest MAE (3.31) which makes sense — it's the simplest line (two
stations, peak-only service, no branches). The delay pattern is basically bimodal: zero when
closed, small positive value during peak.

**Piccadilly** (15.75) and **Metropolitan** (15.21) are by far the worst. Both have complex
routes with long suburban sections and wide delay distributions. A single signal failure near
central stations creates cascading delays that take multiple intervals to clear, and the model
hasn't seen enough of these spike events to predict them well.

**Victoria** (9.60) is in between. Despite being a simple line, its high ridership at major
interchanges (Victoria, King's Cross, Brixton) means small signalling problems create big
delay cascades. The model doesn't have station-level crowding data to distinguish these cases.

The ARIMA baseline analysis (`analysis/arima_baseline.py`) reveals a more significant finding:
a univariate SARIMA(1,1,1)(1,1,1,24) model — which uses only the delay time series itself, with no
access to weather, network state, or temporal features — outperforms LightGBM on 7 of 11 lines:

| Line | ARIMA MAE | LightGBM MAE | Winner |
|---|---|---|---|
| Bakerloo | 0.916 | 4.125 | ARIMA |
| Circle | 0.717 | 4.595 | ARIMA |
| Hammersmith & City | 0.580 | 4.180 | ARIMA |
| Jubilee | 0.720 | 3.953 | ARIMA |
| Northern | 1.524 | 4.226 | ARIMA |
| Waterloo & City | 0.751 | 3.308 | ARIMA |
| Central | 1.960 | 4.137 | ARIMA |
| Victoria | 15.343 | **9.597** | LightGBM |
| Metropolitan | 17.157 | **15.214** | LightGBM |
| Piccadilly | 17.634 | **15.753** | LightGBM |
| District | 19,341* | 4.262 | LightGBM |

*District SARIMA diverged numerically; ARIMA(1,1,1) fallback applied but still unstable at N=54
training rows.

The takeaway is that on well-behaved lines with moderate delay variance, the lag autocorrelation
alone provides enough signal, and LightGBM's extra features just add noise at this scale. On
high-variance lines, ARIMA also fails, but LightGBM handles the occasional outlier spike better
than ARIMA's difference operators. The exogenous features (weather, network, temporal flags) will
probably start earning their keep once there's about 5,000+ rows per line.

**Learning curves** (`analysis/learning_curves.py`) confirm the high-variance diagnosis:

| Proportion | N train | Train MAE | Val MAE |
|---|---|---|---|
| 10% | 59 | 5.374 | 8.606 |
| 30% | 179 | 2.166 | 4.781 |
| 50% | 299 | 1.702 | 4.653 |
| 80% | 478 | 1.658 | 4.555 |
| 100% | 598 | 1.729 | 4.638 |

The gap at full data (1.73 vs 4.64, difference of 2.91 min) hasn't closed from 50% to 100% of
the data — it's basically flat. The validation MAE is still declining at the right edge of the
curve, so more data would definitely help. It would reduce overfitting on the exogenous features
and let the network/weather groups contribute their actual signal.

---

## 3. Regression vs Ordinal Classification

Regression was chosen over ordinal classification for a few reasons:

**Continuous output is more useful.** The point of this system is to give passengers a delay
estimate in minutes. MAE in minutes makes intuitive sense ("predictions are off by ~2 minutes on
average") and maps directly to passenger decisions ("is it worth waiting?"). Ordinal classification
would force you to bin delays into categories like {Good, Minor, Severe}, losing the ability to
distinguish a 2-minute delay from a 9-minute one — both would be "Minor Delays".

**Keeps more information.** Converting delay minutes to a label before training throws away
variation. A regression model can learn the category boundaries as emergent behaviour; an
ordinal classifier assumes them upfront.

**Simpler to implement.** LightGBM's regression objective (L2 with MAE eval) is standard and
well-understood. Ordinal regression in tree models requires either specialist implementations
or softmax proxies, both adding complexity.

**The counter-argument** is that passengers really care about thresholds, not exact minutes.
The difference between 1 and 2 minutes doesn't matter; the difference between 4 and 6 minutes
(crossing the "Minor Delays" threshold) actually changes decisions. Ordinal classification would
be trained to get these threshold crossings right. It would also give calibrated probabilities
("60% chance of Minor Delays") rather than a point estimate plus CI.

A hybrid approach — regression for the estimate, ordinal calibration for the status label — would
be a natural next step.

---

## 4. Risky Assumptions

**Stationarity.** The dataset covers about one week in early March 2026. The assumption that this
generalises to other weeks/seasons is quite strong. Summer heat causes signal failures on deep
tube lines; September has a back-to-school crowding surge; December has a completely different
pattern. To validate, you'd want data from at least 3 different seasons, then check whether
a model trained on weeks 1–3 holds up on week 4. A KPSS stationarity test on the autocorrelation
structure would be a more formal check.

**Independence.** Each row is treated as independent given its features, but delays on different
lines at the same timestamp are clearly correlated — they share passenger demand and feed into
each other's network features. The leave-one-out features partially address this, but there's
still residual dependence. Computing Moran's I across lines would show whether the residuals
are still correlated, in which case a multi-output model might be better.

**Weather data.** OpenWeatherMap gives a single reading for central London, but weather varies
across the network — Heathrow vs Wimbledon can be quite different. You'd want to compare against
Met Office ASOS stations at multiple points to check whether the single-point approximation holds.

**Labels.** TfL delay figures come through the unified API with a 1–5 minute reporting lag, so
training labels may underestimate delays for newly emerging disruptions. Comparing against TfL's
monthly retrospective performance data would quantify this.

---

## 5. Ethical Considerations

**False alarms.** If TfL operationally used this system, false positives (predicting Severe Delays
when service is fine) could trigger unnecessary bus diversions, incident protocols, and passenger
announcements. Repeated false alarms would erode trust and waste resources. A confusion matrix over
the status categories would be needed before any real deployment.

**Feedback loops.** The model was trained on a world where human controllers made dispatch decisions
without ML. If the model's predictions started influencing those decisions, the outcomes would
change — and that changed data would re-enter training. This is basically Goodhart's Law:
once a measure becomes a target, it stops being a good measure. Managing this would need a
causal evaluation framework and periodic retraining.

**Equity.** Different tube lines serve very different communities. If the model systematically
over-predicts delays on lines serving lower-income areas (maybe due to data representation or
more volatile delay patterns), resource allocation would be biased against passengers who most
depend on public transport. Haven't tested whether per-line MAE correlates with deprivation
indices yet — this should be done before deployment.

**Transparency.** If passengers see "your journey will be delayed by 4 minutes", they should
probably know whether that came from a human controller or an ML model. The EU AI Act's
transparency requirements for AI in transport are relevant here.

---

## 6. Lessons Learned

**Data collection was harder than expected.** Building a reliable collector against two APIs with
different rate limits and auth schemes took more effort than planned. The TfL API only gives
qualitative status strings, not numeric delays, requiring a separate estimation heuristic.
Should have prototyped the collection pipeline earlier and allocated more time for data gathering.

**Synthetic-to-real gap.** The initial synthetic data was unrealistically clean — tighter
distributions, fewer extremes, less temporal autocorrelation than real data. When real data came
in, some feature engineering assumptions broke. Would have been better to use synthetic data only
for pipeline testing and wait for real data before tuning anything.

**Adding features without re-running hyperparameter search was a mistake.** The improvement
history analysis shows this clearly: 37 features with fixed hyperparameters gives MAE 4.638,
while the original 21 features with the same hyperparameters gives 4.426 — the bigger feature
set is actually *worse*. The production model's 2.01 MAE came from hyperparameters tuned
specifically for 21 features. Adding 16 more features without re-searching meant the
regularisation wasn't right for the new feature space.

**Overfitting is worse than the headline numbers suggest.** Learning curves show train MAE 1.73
vs val MAE 4.64 at full data — a 2.91 gap that didn't close from 50% to 100% data. The 37
exogenous features are the main vehicle for this overfitting. Should probably reduce to the 8–10
most important features until the dataset reaches ~5,000 rows, then re-introduce the rest.

**Hyperparameter search was too shallow.** RandomizedSearchCV with n_iter=20 covers less than
0.1% of the parameter space. Bayesian optimisation (Optuna) would have mapped the space more
efficiently. The result is a model that works well overall but whose per-line performance is
fragile — ARIMA beats it on 7 of 11 lines.

---

## 7. Future Work

**Extend to buses and surface transport.** The same pipeline could work for bus routes, Elizabeth
line, and London Overground without major changes. The challenge with buses is the sheer number
of routes (700+), GPS noise, and sensitivity to surface traffic that the current features don't
capture.

**Better data sources.** Three things would most improve accuracy: (1) TfL's Trackernet API for
actual train positions/speeds rather than operationally-reported status; (2) rolling stock age
and maintenance schedules from TfL FOI data; (3) TfL's planned engineering works calendar for
advance knowledge of track closures.

**Concept drift monitoring.** A production system would need monitoring — rolling 7-day MAE with
alerts if it exceeds 1.5x baseline, monthly automated retraining on the previous 90 days, and
A/B shadow testing before promoting new models. Feature drift detection (Population Stability
Index) would catch data quality issues early.
