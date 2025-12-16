import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
import sys

# 设置随机种子以获得可重复的结果
#SEED = 42


# 设置随机种子以获得可重复的结果
SEED = np.random.randint(1, 10000)
print(f"Random Seed: {SEED}")

# 设置TensorFlow和NumPy的随机种子
tf.random.set_seed(SEED)
np.random.seed(SEED)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"densenet169_guava_training_{current_time}.txt"

# 创建日志文件
log_file = open(log_filename, "w")
sys.stdout = log_file
sys.stderr = log_file

print(f"Random Seed: {SEED}")

# 目录
train_dir = "/root/Cropper/Guava/Train"
val_dir = "/root/Cropper/Guava/Valid"
test_dir = "/root/Cropper/Guava/Test"

# 配置 - 使用更合理的设置
# DenseNet的原始论文使用224×224，这是标准设置
# The proposed models were tested with to enhance the parameters of the network to
# attain better results. The models were run for 30 epochs with a batch size of 15, learning
# rate 0.00001, and used Adam optimizer [50] because of better memory usage and better efficiency.
# Softmax was used as the activation function in the last layer [22] which
# returns 0 or 1 values. Categorical cross entropy was used to calculate the loss values of
# all the models.
# Weights and Biases [7] platform was also used for better comparisons to track and
# record the results produced by every model.
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 15  # 论文明确使用15
EPOCHS = 30  # 论文明确使用30
LR = 0.00001  # 论文明确使用0.00001

# 番石榴病害类别
CLASS_LABELS = [
    'Disease_Free_Leaf',
    'Disease_Free_Fruit',
    'Phytophora',
    'Red_Rust',
    'Scab',
    'Styler_and_Root'
]
NUM_CLASSES = len(CLASS_LABELS)

CONFIG = dict(
    SEED=SEED,
    IMG_HEIGHT=IMG_HEIGHT,
    IMG_WIDTH=IMG_WIDTH,
    BATCH_SIZE=BATCH_SIZE,
    EPOCHS=EPOCHS,
    LR=LR,
    CLASS_LABELS=CLASS_LABELS,
    NUM_CLASSES=NUM_CLASSES,
)

print("=== INPUT IMAGES ===")

# 数据预处理 - 基于论文描述使用合理的默认值
print("=== PREPROCESSING ===")
print("Dataset: rescaled, resize, rotated, zoomed, shifted horizontally and vertically")
print("Augmentation done to prevent overfitting")

# 使用更保守的数据增强参数
# # Not clear
# With this data set, we first resized the images for the CNN models. Then, the image has
# been rotated at different angles. The images have also been horizontally flipped as their
# identity is not affected by it. Scaling operations have also been used and the images have
# also been zoomed appropriately Fig
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,  # 适度的旋转
    width_shift_range=0.1,  # 适度的水平平移
    height_shift_range=0.1,  # 适度的垂直平移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.1,  # 适度的缩放
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 数据生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED,
    classes=CLASS_LABELS
)

valid_generator = valid_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED,
    classes=CLASS_LABELS
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED,
    classes=CLASS_LABELS
)

print(f"Found {train_generator.samples} training images")
print(f"Found {valid_generator.samples} validation images")
print(f"Found {test_generator.samples} test images")

print("=== PRE-TRAINED MODEL: DenseNet169 ===")


def create_densenet169_model():
    """
    按照论文流程图的架构创建DenseNet169模型：
    Pre-trained Models → Average Pooling → Flatten → Fully connected Layer → FC with Softmax
    """
    print("=== MODIFIED LAYERS ===")
    print("1. Average Pooling")
    print("2. Flatten")
    print("3. Fully connected Layer")
    print("4. FC with Softmax")

    # 创建DenseNet169基础模型 - 使用224×224输入
    base_model = tf.keras.applications.DenseNet169(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling=None  # 不在基础模型中包含池化层
    )

    # 按照流程图构建模型架构
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(name='average_pooling'),  # Average Pooling
        tf.keras.layers.Flatten(name='flatten'),  # Flatten
        tf.keras.layers.Dense(256, activation='relu', name='fully_connected_layer'),  # Fully connected Layer
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='fc_with_softmax')  # FC with Softmax
    ])

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)    # Not clear

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # Clear
        metrics=['accuracy']
    )

    print(f"Created DenseNet169 with {model.count_params():,} parameters")
    model.summary()

    return model


# 创建DenseNet169模型
model = create_densenet169_model()

print('Training configuration: ', CONFIG)

print("=== TRAINING AND TUNING HYPERPARAMETERS ===")
print(f"Training DenseNet169 for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print(f"Learning rate: {LR}")

# 训练模型 - 按照论文直接训练30个epoch，不使用回调
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    verbose=1
)

# 保存训练历史
pd.DataFrame(history.history).to_csv('densenet169_training_history.csv', index=False)

# 评估模型
print("=== CALCULATIONS OF ACCURACY, PRECISION, RECALL, F1-SCORE, CONFUSION MATRIX ===")

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 预测
preds = model.predict(test_generator)
y_preds = np.argmax(preds, axis=1)
y_test = np.array(test_generator.labels)

# 计算指标
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_preds, average='macro', zero_division=0)
recall = recall_score(y_test, y_preds, average='macro', zero_division=0)
f1 = f1_score(y_test, y_preds, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test, y_preds)

print("=== FIND OUT BEST MODEL ===")
print(f"Best Model: DenseNet169 with accuracy {test_accuracy:.4f}")

# 混淆矩阵
cm_data = confusion_matrix(y_test, y_preds)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index=CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'

plt.figure(figsize=(12, 10))
plt.title('Confusion Matrix - DenseNet169', fontsize=16)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            annot_kws={"size": 12}, linewidths=0.5)
plt.tight_layout()
plt.savefig('densenet169_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 分类报告
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_preds, target_names=CLASS_LABELS))

# 特异性计算
specificities = []
for i in range(NUM_CLASSES):
    tn = np.sum(np.delete(np.delete(cm_data, i, axis=0), i, axis=1))
    fp = np.sum(cm_data[:, i]) - cm_data[i, i]
    specificities.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
specificity = np.mean(specificities)

# 输出最终结果
print("\n" + "=" * 60)
print("FINAL TEST PERFORMANCE METRICS - DENSENET169")
print("=" * 60)
print(f"• Test Accuracy:     {test_accuracy:.4f}")
print(f"• Test Precision:    {precision:.4f} (macro avg)")
print(f"• Test Recall:       {recall:.4f} (macro avg)")
print(f"• Test F1 Score:     {f1:.4f} (macro avg)")
print(f"• Test Specificity:  {specificity:.4f}")
print(f"• Test MCC:          {mcc:.4f}")
print("=" * 60)

# 保存模型
model.save('densenet169_guava_model.h5')
print("Model saved as: densenet169_guava_model.h5")

# 绘制训练历史
plt.figure(figsize=(15, 5))

# 准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - DenseNet169')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - DenseNet169')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('densenet169_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存结果到CSV
results_df = pd.DataFrame({
    'Model': ['DenseNet169'],
    'Test_Accuracy': [test_accuracy],
    'Test_Loss': [test_loss],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1],
    'Specificity': [specificity],
    'MCC': [mcc]
})
results_df.to_csv('densenet169_results.csv', index=False)

print(f"\nTraining completed successfully!")
print(f"Results saved to: densenet169_results.csv")
print(f"Training history saved to: densenet169_training_history.csv")
print(f"Model saved as: densenet169_guava_model.h5")

log_file.close()

# 恢复标准输出
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print(f"Training completed! Check {log_filename} for details.")
print(f"DenseNet169 achieved test accuracy: {test_accuracy:.4f}")