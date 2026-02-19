from train import train

def main():
    print("INITIATING TRAINING FOR ALL MODELS...")

    models = ['ae', 'cnn', 'mlp_clip', 'mlp_txt']

    for model in models:
        print(f"\n\n========================================")
        print(f"STARTING: {model.upper()}")
        print(f"========================================")
        
        train(model)
        
        print(f"✅ {model.upper()} FINISHED.")

    print("\n\nALL MODELS TRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    main()