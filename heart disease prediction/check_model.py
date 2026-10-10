import os
import joblib

for fn in sorted(os.listdir('.')):
    if not fn.endswith('.joblib'):
        continue
    print('FILE', fn, os.path.getsize(fn))
    try:
        model = joblib.load(fn)
        print('TYPE', type(model))
        for attr in ['classes_', 'feature_names_in_', 'n_features_in_']:
            print(attr, hasattr(model, attr))
        if hasattr(model, 'classes_'):
            print('CLASSES', list(model.classes_))
        if hasattr(model, 'feature_names_in_'):
            print('FEATURES', list(model.feature_names_in_))
    except Exception as e:
        print('LOAD_ERR', type(e).__name__, e)
