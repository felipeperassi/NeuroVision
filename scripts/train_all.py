from train import train

def train_all():
    print("INITIATING TRAINING FOR ALL MODELS...")

    models = ['MLP', 'CNN_VAE', 'CNN_VGG']

    for model in models:
        print(f"\n\n========================================")
        print(f"STARTING: {model.upper()}")
        print(f"========================================")
        
        train(model)
        
        print(f"✅ {model.upper()} FINISHED.")

    print("\n\nALL MODELS TRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    train_all()