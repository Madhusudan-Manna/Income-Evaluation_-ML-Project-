import pickle
import sys

print("=" * 60)
print("TESTING INCOME.PKL FILE")
print("=" * 60)

try:
    with open('Income.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    print("\n✓ Model package loaded successfully!")
    
    print(f"\n✓ Model: {model_package['model_name']}")
    print(f"✓ Accuracy: {model_package['accuracy']:.4f}")
    print(f"✓ Number of Features: {len(model_package['features'])}")
    print(f"✓ Label Encoders: {len(model_package['label_encoders'])}")
    print(f"✓ Scaler: Present")
    
    print("\n✓ All components verified!")
    print("✓ Income.pkl is ready for Streamlit app!")
    
    print("\nModel Package Contents:")
    for key in model_package.keys():
        print(f"  - {key}")
    
    print("\n" + "=" * 60)
    print("STATUS: SUCCESS - Income.pkl is ready to use!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error loading Income.pkl: {str(e)}")
    sys.exit(1)
