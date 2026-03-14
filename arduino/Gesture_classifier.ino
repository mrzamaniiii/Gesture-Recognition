#include "Arduino_BMI270_BMM150.h"
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <math.h>

constexpr int NUM_SAMPLES = 128;
constexpr int NUM_AXES = 6;
constexpr int NUM_FEATURES = 42;
constexpr int NUM_CLASSES = 4;

const float ACC_THRESHOLD = 1.0f;

const char* CLASS_NAMES[NUM_CLASSES] = {
  "circle",
  "left_right",
  "rest",
  "up_down"
};

const float SCALER_MEAN[NUM_FEATURES] = {
 0.19942442859077503, 0.12001400083618781, 0.3932947434602291, 0.01879978470328286, 0.40692137360690145, 3.783218503937008, 0.008004962927006498, -0.17372557768363536, 0.10652600302302286, 0.6760707878100678, -0.4235565610867962, 0.054761990199899524, 5.825541338582677, 0.0042813348958576376, -0.07721831095514922, 0.12141783775644392, 0.33801349617658166, -0.32804857845616153, 0.19443639745756133, 6.889763779527559, 0.005204088450019893, -1.7698592903572832, 17.221534807616333, 18.15953020845342, -40.75994798282939, 34.87414087038341, 7.8924704724409445, 95.04524842130138, -0.21939262712473329, 15.127008326410309, 15.893010690690964, -29.78155002490742, 33.85792991917903, 5.720964566929134, 138.8715004665909, -1.8842176658286545, 29.12033357054699, 31.408926651937755, -51.70230155760848, 60.27557723353228, 2.4483267716535435, 522.8875270982046
};

const float SCALER_SCALE[NUM_FEATURES] = {
0.42722454520027614, 0.12860505911090414, 0.3139311789821978, 0.5064626729567807, 0.40078199522598656, 4.228155067647704, 0.013595422440722957, 0.7451474147259302, 0.09980705309731171, 0.38686272713694875, 0.9200743786381462, 0.6199076954825322, 5.889139383175801, 0.009874800990509577, 0.3601975288860894, 0.09552451738815895, 0.21288266216539006, 0.5109183490238087, 0.27604014753276707, 6.203240223066164, 0.01532816287651507, 7.62761048194714, 12.636686945664955, 13.704430375908712, 34.002546533259746, 26.488818998113963, 7.858094466585509, 141.69605543363505, 5.857593295215824, 16.764745362957484, 17.077904254321904, 31.736659000875417, 41.47910782483897, 5.637836923144635, 1265.0427522646771, 16.562892859868832, 33.99583959229484, 35.987080126146694, 56.76380264386035, 73.6935882845588, 2.2148403155837793, 1104.3830593040152
};

float sampleBuffer[NUM_SAMPLES][NUM_AXES];

tflite::AllOpsResolver resolver;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model_tflite = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensorArenaSize = 16 * 1024;
uint8_t tensorArena[tensorArenaSize];

float computeMean(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i];
  return s / n;
}

float computeStd(const float* x, int n, float meanVal) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = x[i] - meanVal;
    s += d * d;
  }
  return sqrtf(s / n);
}

float computeRMS(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return sqrtf(s / n);
}

float computeMin(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] < m) m = x[i];
  }
  return m;
}

float computeMax(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > m) m = x[i];
  }
  return m;
}

void computeFrequencyFeatures(const float* x, int n, float fs, float& dominantFreq, float& peakPower) {
  const int maxBin = 16;
  peakPower = 0.0f;
  dominantFreq = 0.0f;

  for (int k = 1; k <= maxBin; k++) {
    float realPart = 0.0f;
    float imagPart = 0.0f;

    for (int t = 0; t < n; t++) {
      float angle = 2.0f * PI * k * t / n;
      realPart += x[t] * cosf(angle);
      imagPart -= x[t] * sinf(angle);
    }

    float power = realPart * realPart + imagPart * imagPart;
    if (power > peakPower) {
      peakPower = power;
      dominantFreq = (fs * k) / n;
    }
  }
}

void extractFeatures(float features[NUM_FEATURES]) {
  float axisSignal[NUM_SAMPLES];
  int idx = 0;

  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
      axisSignal[i] = sampleBuffer[i][axis];
    }

    float meanVal = computeMean(axisSignal, NUM_SAMPLES);
    float stdVal = computeStd(axisSignal, NUM_SAMPLES, meanVal);
    float rmsVal = computeRMS(axisSignal, NUM_SAMPLES);
    float minVal = computeMin(axisSignal, NUM_SAMPLES);
    float maxVal = computeMax(axisSignal, NUM_SAMPLES);

    float dominantFreq, peakPower;
    computeFrequencyFeatures(axisSignal, NUM_SAMPLES, 100.0f, dominantFreq, peakPower);

    features[idx++] = meanVal;
    features[idx++] = stdVal;
    features[idx++] = rmsVal;
    features[idx++] = minVal;
    features[idx++] = maxVal;
    features[idx++] = dominantFreq;
    features[idx++] = peakPower;
  }
}

void normalizeFeatures(float features[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    if (SCALER_SCALE[i] != 0.0f) {
      features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
  }
}

int argmaxOutput() {
  int best = 0;
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > output->data.f[best]) {
      best = i;
    }
  }
  return best;
}

bool waitForMotionTrigger() {
  float aX, aY, aZ;

  while (true) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      if (aSum >= ACC_THRESHOLD) {
        delay(50);
        return true;
      }
    }
  }
}

bool captureWindow() {
  float aX, aY, aZ, gX, gY, gZ;
  int samplesRead = 0;

  while (samplesRead < NUM_SAMPLES) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      sampleBuffer[samplesRead][0] = aX;
      sampleBuffer[samplesRead][1] = aY;
      sampleBuffer[samplesRead][2] = aZ;
      sampleBuffer[samplesRead][3] = gX;
      sampleBuffer[samplesRead][4] = gY;
      sampleBuffer[samplesRead][5] = gZ;

      samplesRead++;
    }
  }

  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Starting final gesture classifier...");

  if (!IMU.begin()) {
    Serial.println("IMU failed");
    while (1);
  }

  model_tflite = tflite::GetModel(gesture_model_tflite);

  static tflite::MicroInterpreter static_interpreter(
      model_tflite,
      resolver,
      tensorArena,
      tensorArenaSize,
      error_reporter,
      nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready!");
}

void loop() {
  if (!waitForMotionTrigger()) return;
  if (!captureWindow()) return;

  float features[NUM_FEATURES];
  extractFeatures(features);
  normalizeFeatures(features);

  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  int pred = argmaxOutput();

  Serial.print("Prediction: ");
  Serial.println(CLASS_NAMES[pred]);

  Serial.print("circle: ");
  Serial.print(output->data.f[0], 6);
  Serial.print(" left_right: ");
  Serial.print(output->data.f[1], 6);
  Serial.print(" rest: ");
  Serial.print(output->data.f[2], 6);
  Serial.print(" up_down: ");
  Serial.println(output->data.f[3], 6);

  Serial.println();
  delay(1000);
}