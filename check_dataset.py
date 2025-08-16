import os

# Define the extracted dataset folder path
extract_path = r"C:\Users\jatin\LymphoScan_Project\data\extracted"

print("Checking dataset file structures...\n")

# Loop through each dataset
for dataset in os.listdir(extract_path):
    dataset_path = os.path.join(extract_path, dataset)
    print(f"📂 {dataset}:")

    if os.path.isdir(dataset_path):
        for subfolder in os.listdir(dataset_path):
            subfolder_path = os.path.join(dataset_path, subfolder)
            print(f"  📁 {subfolder}:")
            
            if os.path.isdir(subfolder_path):
                files = os.listdir(subfolder_path)
                if len(files) > 10:  # Print only first 10 files if too many
                    print(f"    Total files: {len(files)}")
                    print("    First 10 files:", files[:10])
                else:
                    print("    Files:", files)
            else:
                print("    ❌ Not a valid directory!")

    print("\n" + "-" * 50 + "\n")
