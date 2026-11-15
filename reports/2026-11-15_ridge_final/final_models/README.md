# Final models (frozen)

Fitted on all labelled beta images, with lambda chosen by grouped cross-validation on those same images. Their performance must not be reported on the beta images (that would leak validation into test); it is measured once, on the external test set. The cross-validated estimate of what to expect is in ../summary.md.

Load with `joblib.load(path)`; `model.predict_proba(X)` takes the Directional Scores of `all_question_ids` in that order (the selection step inside the model keeps `question_ids`).
