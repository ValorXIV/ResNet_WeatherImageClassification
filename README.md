# การ Finetune Model ResNet เพื่อจำแนกสภาพอากาศจากภาพ

**Repository Link:** [https://github.com/ValorXIV/ResNet_WeatherImageClassification](https://github.com/ValorXIV/ResNet_WeatherImageClassification)

---

## หัวข้อนี้น่าสนใจอย่างไร ทำไมถึงเลือกหัวข้อนี้มาทำเป็น Final Project

จากสภาวะแวดล้อมในปัจจุบันที่โลกของเราร้อนขึ้นเรื่อย ๆ ทำให้สภาพอากาศแปรปรวนและมีการเปลี่ยนแปลงอยู่บ่อยครั้ง ประกอบกับการที่ในปัจจุบัน Social Media เป็นที่นิยมในวงกว้าง และผู้คนต่างอัปโหลดข้อมูลรูปภาพจำนวนมหาศาลลงในอินเทอร์เน็ต  
ทำให้กลุ่มของเราต้องการจะสร้าง **Model ที่ช่วยจำแนกสภาพอากาศจากภาพถ่ายต่าง ๆ** บนอินเทอร์เน็ตขึ้นมา  
เพื่อศึกษาและทำนายสภาพอากาศในแต่ละพื้นที่ และสามารถนำ Model ไปพัฒนาต่อยอดเพื่อการรายงานสภาพอากาศแบบ Real-time จากภาพถ่ายในแต่ละบริเวณได้อีกด้วย

---

## ทำไมหัวข้อนี้จึงต้องใช้ Deep Learning

ภาพสภาพอากาศในแต่ละวันมีความต่อเนื่องและเปลี่ยนแปลงบ่อย อีกทั้งมีจำนวนมหาศาลจากหลายพื้นที่  
การจำแนกภาพด้วยวิธีอื่น ๆ เช่น การจำแนกด้วยฝีมือมนุษย์นั้นทำได้ช้าและสิ้นเปลืองทรัพยากร  
แต่หากใช้ **Deep Learning Model** จะสามารถจำแนกภาพได้อย่างรวดเร็วและต่อเนื่องตลอดเวลา  
นอกจากนี้ยังสามารถเรียนรู้ลักษณะของภาพแต่ละสภาพอากาศได้อย่างถูกต้องด้วยข้อมูลจำนวนมาก

---

## สถาปัตยกรรม Deep Learning ที่ใช้

ใช้ **Model ResNet18** ซึ่งเป็น **Convolutional Neural Network (CNN)** ที่ผ่านการ **Pre-train** แล้ว  
นำมาเชื่อมต่อกับ **Feedforward Neural Network (FNN)** เพื่อ **Finetune** กับ Dataset ใหม่

- **Output จาก ResNet18:** ขนาด 512
- **Layer ใหม่ที่ต่อเพิ่ม:**
  - Hidden Layer: 256 Nodes, Activation = ReLU, Dropout = 0.5
  - Output Layer: 6 Nodes (แทน 6 Classes ของสภาพอากาศ)

ผลลัพธ์คือโมเดลที่สามารถปรับตัวกับ Dataset ใหม่ได้ดียิ่งขึ้นและลดการ Overfitting

---

## อธิบายโค้ด PyTorch

### 1. การเตรียมข้อมูล
- ปรับขนาดรูปภาพเป็น **224x224 พิกเซล**
- ทำ **Data Augmentation** เพื่อเพิ่มความหลากหลายของข้อมูล
- แบ่งข้อมูลเป็น 2 ชุด:
  - Training Set (80%)
  - Validation Set (20%)
```python
```
### 2. การสร้างโมเดล
- ดาวน์โหลด **ResNet18** ที่ผ่านการ Pre-train
- **Freeze Parameters** เดิมเพื่อไม่ให้ถูกอัปเดตระหว่าง Train
- นำ Feature จาก Layer สุดท้ายของ ResNet18 มาต่อกับ FNN ที่มี:
  - Hidden Layer: 256 Nodes + Dropout
  - Output Layer: 6 Nodes

### 3. การกำหนด Loss และ Optimizer
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** (ตามที่ใช้ในโค้ด)

### 4. การเทรนโมเดล
- คำนวณ Loss และ Gradient
- อัปเดต Weight ตามค่า Gradient
- ทำซ้ำตามจำนวน Epoch = **7**

### 5. การสรุปผล
- แสดง **Training & Validation Loss**
- แสดง **Accuracy**
- ใช้เพื่อปรับค่า Parameter และสรุปผลการเรียนรู้

---

## Dataset ที่ใช้

**Dataset:** [Multiclass Weather Dataset – Kaggle](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset)

- รวมภาพของสภาพอากาศต่าง ๆ ภายใต้สัญญา Creative Commons
- มีภาพจากทั่วโลก โดยแต่ละภาพแทน 1 ใน 5 ประเภทสภาพอากาศ:
  - Cloudy (มีเมฆมาก)
  - Foggy (มีหมอก)
  - Rainy (มีฝน)
  - Shine (มีแสงแดด)
  - Sunrise (พระอาทิตย์ขึ้น/ตก)
- ใช้เพื่อสร้างโมเดลจำแนกภาพสภาพอากาศให้เป็นหนึ่งใน 5 ประเภทข้างต้น

---

## วิธีการ Train Model

1. ปรับขนาดข้อมูลเป็น 224x224 พิกเซล  
2. ทำ Data Augmentation  
3. แบ่งข้อมูลเป็น Training และ Validation Sets  
4. Finetune ResNet18 ด้วย Layer ใหม่  
5. กำหนด Learning Rate, Loss Function, Optimizer  
6. Train Model โดยทำการคำนวณ Loss, Gradient และปรับ Weight  
7. วัดผลด้วย Validation Set

---

## การประเมินผล (Evaluation)

- จาก **Validation Loss** และ **Accuracy** พบว่า:
  - Loss ลดจาก ~1.15 → ~0.4  
  - Accuracy เพิ่มจาก 0.73 → 0.92  
  → แสดงว่า Model ทำงานได้ดีและไม่ Overfit

- จาก **Classification Report** และ **Confusion Matrix**:
  - Accuracy = 92%
  - F1-score (ทุก Class ยกเว้นข้อมูลเสีย): 0.87–0.96  
  ⇒ โมเดลมีความแม่นยำสูงและทำงานได้ตรงตามเป้าหมาย

---

## สมาชิกในกลุ่ม

**นายชญานนท์ มานะกิจจานนท์ (6510503298)**  
- ค้นหาข้อมูลและ Dataset  
- ออกแบบโครงสร้าง Model  
- จัดทำรายงาน  
**สัดส่วนงาน:** 50%

**นายศุภกิตต์ วงศ์โต (6510503816)**  
- สร้าง Model ตามแผน  
- Train Model ให้ได้ผลลัพธ์ตามเป้าหมาย  
- จัดทำรายงาน  
**สัดส่วนงาน:** 50%
