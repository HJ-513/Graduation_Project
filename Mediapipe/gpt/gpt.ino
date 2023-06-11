#include <Servo.h>

// 각 서보모터의 핀 번호
const int thumb_pin = 2;
const int index_pin = 3;
const int middle_pin = 4;
const int ring_pin = 5;
const int pinky_pin = 6;
const int wrist_x_pin = 7;
const int wrist_y_pin = 8;

// 각 서보모터 객체
Servo thumb_servo;
Servo index_servo;
Servo middle_servo;
Servo ring_servo;
Servo pinky_servo;
Servo wrist_x_servo;
Servo wrist_y_servo;

// 손가락의 각도
int thumb_angle = 0;
int index_angle = 0;
int middle_angle = 0;
int ring_angle = 0;
int pinky_angle = 0;

// 손목의 각도
int wrist_x_angle = 0;
int wrist_y_angle = 0;

// 시리얼 통신 버퍼 크기
const int buffer_size = 32;

void setup() {
  // 시리얼 통신 시작
  Serial.begin(9600);

  // 서보모터 핀 모드 설정
  thumb_servo.attach(thumb_pin);
  index_servo.attach(index_pin);
  middle_servo.attach(middle_pin);
  ring_servo.attach(ring_pin);
  pinky_servo.attach(pinky_pin);
  wrist_x_servo.attach(wrist_x_pin);
  wrist_y_servo.attach(wrist_y_pin);
}

void loop() {
  // 시리얼 버퍼에서 데이터 읽기
  char buffer[buffer_size];
  int idx = 0;
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      buffer[idx] = '\0';
      break;
    }
    buffer[idx++] = c;
  }

  // 데이터를 콤마 단위로 분리
  char* ptr;
  ptr = strtok(buffer, ",");
  while (ptr != NULL) {
    int index = atoi(ptr);

    switch (index) {
      case 4: // 엄지
        ptr = strtok(NULL, ",");
        thumb_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        thumb_servo.write(thumb_angle);
        break;

      case 8: // 검지
        ptr = strtok(NULL, ",");
        index_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        index_servo.write(index_angle);
        break;

      case 12: // 중지
        ptr = strtok(NULL, ",");
        middle_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        middle_servo.write(middle_angle);
        break;

      case 16: // 약지
        ptr = strtok(NULL, ",");
        ring_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        ring_servo.write(ring_angle);
        break;

      case 20: // 새끼
        ptr = strtok(NULL, ",");
        pinky_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        pinky_servo.write(pinky_angle);
        break;

      case 24: // 손목 X
        ptr = strtok(NULL, ",");
        wrist_x_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        wrist_x_servo.write(wrist_x_angle);
        break;

      case 25: // 손목 Y
        ptr = strtok(NULL, ",");
        wrist_y_angle = map(atof(ptr), 0.0, 1.0, 0, 180);
        wrist_y_servo.write(wrist_y_angle);
        break;

      default:
        ptr = strtok(NULL, ",");
        break;
    }
  }
}
