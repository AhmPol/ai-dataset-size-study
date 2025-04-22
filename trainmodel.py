import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import time
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/ScienceFair"

low_path = f"{base_path}/lowimages"
mid_path = f"{base_path}/midimages"
high_path = f"{base_path}/highimages"
test_path = f"{base_path}/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

low_ds = tf.keras.preprocessing.image_dataset_from_directory(
    low_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

mid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    mid_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

high_ds = tf.keras.preprocessing.image_dataset_from_directory(
    high_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=1,  # 1 so we can track time per image
    shuffle=False
)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)  # 2 classes: Apple, Banana
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

model_low = build_model()
history_low = model_low.fit(low_ds, epochs=15)

model_mid = build_model()
history_mid = model_mid.fit(mid_ds, epochs=15)

model_high = build_model()
history_high = model_high.fit(high_ds, epochs=15)

# Path to test images
base_dir = "/content/drive/MyDrive/ScienceFair"
test_dir = pathlib.Path(base_dir) / 'test'
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=1,
    shuffle=False  # Ensures consistent order
)

# Function to show image and prediction
def show_predictions(model, test_ds):
    class_names = test_ds.class_names

    for images, _ in test_ds:
        prediction = model.predict(images)
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

        plt.imshow(images[0].numpy().astype("uint8"))
        plt.title(f"Predicted: {class_names[predicted_class]}")
        plt.axis("off")
        plt.show()

# Example usage for one model (repeat for others)
print("Low Model Predictions:\n")
show_predictions(model_low, test_ds)

# To test the other models, use:
# show_predictions(model_mid, test_ds)
# show_predictions(model_high, test_ds)

# Test image path
base_dir = "/content/drive/MyDrive/ScienceFair"
test_dir = pathlib.Path(base_dir) / 'test'
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=1  # Evaluate one image at a time
)

# Function to evaluate a model and store detailed results
def evaluate_model(model, model_name, test_ds):
    results = []
    image_num = 0
    correct_predictions = 0
    total_time = 0

    for images, labels in test_ds:
        image_num += 1
        start_time = time.time()
        predictions = model.predict(images, verbose=0)
        end_time = time.time()

        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        actual_class = labels.numpy()[0]
        prediction_time = end_time - start_time
        correct = int(predicted_class == actual_class)

        results.append({
            "Model": model_name,
            "Image Number": image_num,
            "Predicted Class": predicted_class,
            "Actual Class": actual_class,
            "Correct": "Yes" if correct else "No",
            "Prediction Time (s)": round(prediction_time, 4)
        })

        total_time += prediction_time
        correct_predictions += correct

    accuracy = correct_predictions / image_num * 100
    avg_time = total_time / image_num

    print(f"\n🔍 {model_name} Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Wrong Predictions: {image_num - correct_predictions}")
    print(f"Average Prediction Time: {avg_time:.4f}s\n")

    return results, accuracy

# Run evaluations
results_low, accuracy_low = evaluate_model(model_low, "Low", test_ds)
results_mid, accuracy_mid = evaluate_model(model_mid, "Mid", test_ds)
results_high, accuracy_high = evaluate_model(model_high, "High", test_ds)

# Combine all results into a single DataFrame
df_all = pd.DataFrame(results_low + results_mid + results_high)

# Display a preview
df_all.head()

# Export to CSV so you can download it and use in Google Sheets
output_path = "/content/All_Model_Results.csv"
df_all.to_csv(output_path, index=False)
print(f"✅ CSV file saved to: {output_path}")

# Create a bar chart to visualize accuracy for each model
models = ['Low', 'Mid', 'High']
accuracies = [accuracy_low, accuracy_mid, accuracy_high]

plt.figure(figsize=(10,6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison for Different Models')
plt.ylim(0, 100)
plt.show()

# You can also create additional charts for other comparisons (e.g., prediction time per model)
prediction_times = [df_all[df_all["Model"] == "Low"]["Prediction Time (s)"].mean(),
                    df_all[df_all["Model"] == "Mid"]["Prediction Time (s)"].mean(),
                    df_all[df_all["Model"] == "High"]["Prediction Time (s)"].mean()]

plt.figure(figsize=(10,6))
plt.bar(models, prediction_times, color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Average Prediction Time (s)')
plt.title('Average Prediction Time for Different Models')
plt.ylim(0, max(prediction_times) + 0.01)
plt.show()