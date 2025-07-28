"""
MobileNet 레이어 구조 디버깅
"""
import tensorflow as tf

# 새 모델 생성해서 구조 확인
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Lambda(lambda x: x / 127.5)(inputs)
x = tf.keras.layers.Lambda(lambda x: x - 1.0)(x)

mobilenet = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None,
    input_tensor=x
)

print("MobileNet 레이어 구조:")
print(f"MobileNet 이름: {mobilenet.name}")
print(f"MobileNet 레이어 수: {len(mobilenet.layers)}")

print("\n처음 10개 레이어:")
for i, layer in enumerate(mobilenet.layers[:10]):
    print(f"  {i}: {layer.__class__.__name__} - {layer.name}")

print("\n마지막 5개 레이어:")
for i, layer in enumerate(mobilenet.layers[-5:], len(mobilenet.layers)-5):
    print(f"  {i}: {layer.__class__.__name__} - {layer.name}")

# 전체 모델 구조
x = mobilenet.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(f"\n전체 모델 레이어:")
for i, layer in enumerate(model.layers):
    print(f"  {i}: {layer.__class__.__name__} - {layer.name}")
    if hasattr(layer, 'layers'):
        print(f"     -> 서브레이어 {len(layer.layers)}개 포함")