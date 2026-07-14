/*
  ============================================================
  GLYCOSAFE - Simulasi Firmware (Wokwi)
  ============================================================
  Perbaikan dari versi sebelumnya:
  1. Konversi suhu NTC memakai persamaan Beta (bukan linear voltase x 100)
  2. Sinyal EKG (potensiometer, placeholder AD8232) dipakai dalam logika,
     tidak hanya ditampilkan
  3. Filter Moving Average untuk EKG & GSR (mengurangi motion artifact,
     sesuai klaim di proposal Bab 4.6)
  4. State machine 3 level: NORMAL -> WASPADA -> KRITIS
     (menggantikan if-else datar sebelumnya, lebih dekat ke alur
     "Status Kondisi" pada arsitektur software / Gambar 5)
  5. Logika context-aware yang benar sesuai proposal:
     - Motor getar aktif saat pasien BERGERAK AKTIF (getaran senyap)
     - Buzzer keras aktif saat pasien DIAM/TIDUR (butuh alarm keras
       untuk membangunkan / menarik perhatian sekitar)
     IMU dipakai untuk mengklasifikasikan status aktif vs diam,
     BUKAN untuk memicu alarm langsung dari benturan/goncangan.
  6. Fail-safe: jika MPU6050 gagal terkoneksi, sistem tidak crash dan
     tidak salah baca data kosong; fallback ke mode aman.
  7. Hysteresis sederhana: kondisi harus konsisten selama N pembacaan
     berturut-turut sebelum status berubah, supaya alarm tidak flicker
     akibat noise sesaat.
  8. Tempat pemanggilan model TinyML/LSTM ditandai jelas sebagai
     placeholder (runLSTMInference), karena MCU simulasi Wokwi belum
     bisa menjalankan model TensorFlow Lite Micro sungguhan.
     GPS/BLE/MQTT juga ditandai sebagai TODO untuk implementasi
     di hardware asli (di luar cakupan simulator).
  ============================================================
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ------------------------------------------------------------
// PEMETAAN PIN (sesuai diagram.json Wokwi, tidak diubah)
// ------------------------------------------------------------
const int pinEKG        = 1;   // Potensiometer atas -> placeholder AD8232
const int pinGSR         = 2;  // Potensiometer bawah -> placeholder Grove GSR
const int pinNTC         = 3;  // NTC analog -> placeholder MAX30205
const int pinBuzzer      = 5;  // Piezo Buzzer
const int pinMotorGetar  = 6;  // LED merah -> placeholder ERM coin vibration motor

// I2C untuk MPU6050 (IMU)
const int pinSDA = 8;
const int pinSCL = 9;

Adafruit_MPU6050 mpu;
bool mpuTerkoneksi = false;

// ------------------------------------------------------------
// KONSTANTA KONVERSI NTC (persamaan Beta / B-parameter)
// NOTE: Nilai R0, T0, B, R_SERIES di bawah adalah nilai umum untuk
// NTC 10k tipikal (default part Wokwi). Saat pindah ke hardware
// asli dengan MAX30205 (sensor I2C digital), blok konversi NTC ini
// TIDAK dipakai lagi -- cukup panggil library MAX30205 langsung.
// ------------------------------------------------------------
const float NTC_R0        = 10000.0;   // Resistansi NTC pada suhu referensi (ohm)
const float NTC_T0_KELVIN = 298.15;    // Suhu referensi = 25 C dalam Kelvin
const float NTC_BETA      = 3950.0;    // Koefisien Beta khas NTC 10k
const float NTC_R_SERIES  = 10000.0;   // Resistor pembagi tegangan (ohm)
const float ADC_MAX       = 4095.0;    // Resolusi ADC 12-bit ESP32-S3
const float VCC           = 3.3;

// ------------------------------------------------------------
// PARAMETER FILTER & THRESHOLD
// PENTING: Angka-angka ini masih perkiraan awal untuk simulasi.
// Sebelum uji coba dengan sensor & pasien sungguhan, WAJIB
// dikalibrasi ulang berdasarkan data riil (rentang ADC aktual,
// baseline tiap individu, dsb).
// ------------------------------------------------------------
const int JUMLAH_SAMPEL_FILTER = 10;    // ukuran window moving average
const int GSR_THRESHOLD_WASPADA  = 2200;
const int GSR_THRESHOLD_KRITIS   = 3000;
const float EKG_VARIABILITAS_THRESHOLD = 150.0; // proxy sederhana pengganti HRV asli
const float AKTIVITAS_THRESHOLD_G = 1.5;        // ambang magnitudo akselerasi utk "bergerak aktif"
const int JUMLAH_KONSISTEN_UNTUK_UBAH_STATUS = 4; // hysteresis: harus konsisten N loop

// ------------------------------------------------------------
// BUFFER MOVING AVERAGE
// ------------------------------------------------------------
int bufferEKG[JUMLAH_SAMPEL_FILTER];
int bufferGSR[JUMLAH_SAMPEL_FILTER];
int indexBuffer = 0;
bool bufferPenuh = false;

// ------------------------------------------------------------
// STATE MACHINE
// ------------------------------------------------------------
enum StatusKondisi { NORMAL, WASPADA, KRITIS };
StatusKondisi statusSaatIni = NORMAL;
StatusKondisi statusKandidat = NORMAL;
int hitunganKonsisten = 0;

unsigned long waktuTerakhirBaca = 0;
const unsigned long INTERVAL_BACA_MS = 500; // ganti delay() blocking dgn millis()

void setup() {
  Serial.begin(115200);
  delay(200);

  Wire.begin(pinSDA, pinSCL);

  pinMode(pinBuzzer, OUTPUT);
  pinMode(pinMotorGetar, OUTPUT);
  digitalWrite(pinBuzzer, LOW);
  digitalWrite(pinMotorGetar, LOW);

  // Inisialisasi buffer filter
  for (int i = 0; i < JUMLAH_SAMPEL_FILTER; i++) {
    bufferEKG[i] = 0;
    bufferGSR[i] = 0;
  }

  // Inisialisasi MPU6050 dengan penanganan gagal-konek (fail-safe)
  if (!mpu.begin()) {
    Serial.println("[-] Gagal menemukan chip MPU6050. Sistem masuk MODE AMAN.");
    mpuTerkoneksi = false;
  } else {
    Serial.println("[+] MPU6050 Berhasil Terkoneksi.");
    mpuTerkoneksi = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  Serial.println("=== GLYCOSAFE SIMULATION READY ===");
}

// ------------------------------------------------------------
// Fungsi: konversi ADC NTC -> Celsius memakai persamaan Beta
// ------------------------------------------------------------
float konversiNTCkeCelsius(int nilaiADC) {
  // Hindari pembagian dengan nol jika ADC = 0 atau maksimum
  if (nilaiADC <= 0) nilaiADC = 1;
  if (nilaiADC >= ADC_MAX) nilaiADC = ADC_MAX - 1;

  float voltase = nilaiADC * (VCC / ADC_MAX);

  // Hitung resistansi NTC dari pembagi tegangan
  float resistansiNTC = NTC_R_SERIES * (VCC / voltase - 1.0);

  // Persamaan Beta (Simplified Steinhart-Hart)
  float suhuKelvin = 1.0 / ( (1.0 / NTC_T0_KELVIN) +
                              (1.0 / NTC_BETA) * log(resistansiNTC / NTC_R0) );
  float suhuCelsius = suhuKelvin - 273.15;

  return suhuCelsius;
}

// ------------------------------------------------------------
// Fungsi: moving average sederhana
// ------------------------------------------------------------
int hitungRataRata(int buffer[]) {
  long total = 0;
  int n = bufferPenuh ? JUMLAH_SAMPEL_FILTER : (indexBuffer == 0 ? 1 : indexBuffer);
  for (int i = 0; i < n; i++) total += buffer[i];
  return total / n;
}

// ------------------------------------------------------------
// Fungsi: variabilitas sederhana dari buffer EKG (proxy HRV)
// NOTE: Ini BUKAN perhitungan HRV klinis sesungguhnya (SDNN/RMSSD).
// Karena sinyal EKG di simulasi hanya berasal dari potensiometer
// (tidak berbentuk gelombang EKG asli), variabilitas dihitung
// sebagai standar deviasi sederhana dari sampel dalam buffer,
// hanya untuk memastikan EKG ikut berkontribusi ke logika.
// Saat hardware asli (AD8232) tersedia, fungsi ini digantikan
// oleh ekstraksi fitur HRV yang sesungguhnya (deteksi puncak R,
// interval R-R, lalu SDNN/RMSSD).
// ------------------------------------------------------------
float hitungVariabilitasEKG(int buffer[]) {
  int n = bufferPenuh ? JUMLAH_SAMPEL_FILTER : (indexBuffer == 0 ? 1 : indexBuffer);
  float rataRata = hitungRataRata(buffer);
  float jumlahKuadrat = 0;
  for (int i = 0; i < n; i++) {
    float selisih = buffer[i] - rataRata;
    jumlahKuadrat += selisih * selisih;
  }
  return sqrt(jumlahKuadrat / n);
}

// ------------------------------------------------------------
// Placeholder tempat inferensi model TinyML/LSTM dipanggil.
// Di simulator Wokwi, model TensorFlow Lite Micro tidak dijalankan
// (di luar cakupan simulasi hardware ini). Fungsi ini mengembalikan
// status berbasis aturan (rule-based) sebagai pengganti sementara,
// dengan struktur input/output yang sama seperti nanti dipakai model
// LSTM sungguhan, supaya integrasi ke depan tinggal mengganti isi
// fungsi ini tanpa mengubah alur program yang lain.
// ------------------------------------------------------------
StatusKondisi runInferensiStatus(int gsrRataRata, float variabilitasEKG, float suhuC) {
  // TODO: ganti isi fungsi ini dengan pemanggilan model TFLite Micro
  // (interpreter->Invoke()) begitu model LSTM hasil training tersedia.
  if (gsrRataRata >= GSR_THRESHOLD_KRITIS || variabilitasEKG >= EKG_VARIABILITAS_THRESHOLD * 1.5) {
    return KRITIS;
  } else if (gsrRataRata >= GSR_THRESHOLD_WASPADA || variabilitasEKG >= EKG_VARIABILITAS_THRESHOLD) {
    return WASPADA;
  }
  return NORMAL;
}

void loop() {
  unsigned long sekarang = millis();
  if (sekarang - waktuTerakhirBaca < INTERVAL_BACA_MS) {
    return; // non-blocking, gantikan delay(500) sebelumnya
  }
  waktuTerakhirBaca = sekarang;

  // 1. Akuisisi data mentah
  int nilaiEKGmentah = analogRead(pinEKG);
  int nilaiGSRmentah = analogRead(pinGSR);
  int nilaiNTCmentah = analogRead(pinNTC);

  // 2. Masukkan ke buffer moving average
  bufferEKG[indexBuffer] = nilaiEKGmentah;
  bufferGSR[indexBuffer] = nilaiGSRmentah;
  indexBuffer++;
  if (indexBuffer >= JUMLAH_SAMPEL_FILTER) {
    indexBuffer = 0;
    bufferPenuh = true;
  }

  int gsrTerfilter = hitungRataRata(bufferGSR);
  float variabilitasEKG = hitungVariabilitasEKG(bufferEKG);
  float suhuCelsius = konversiNTCkeCelsius(nilaiNTCmentah);

  // 3. Baca IMU (dengan fail-safe)
  float magnitudoAksel = 0.0;
  bool sedangAktifBergerak = false;

  if (mpuTerkoneksi) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Magnitudo akselerasi total dikurangi gravitasi (~9.8 m/s^2)
    // dipakai untuk mengklasifikasi status gerak, BUKAN sebagai
    // pemicu alarm langsung dari goncangan/benturan.
    float ax = a.acceleration.x;
    float ay = a.acceleration.y;
    float az = a.acceleration.z;
    magnitudoAksel = sqrt(ax * ax + ay * ay + az * az) - 9.8;
    if (magnitudoAksel < 0) magnitudoAksel = -magnitudoAksel;

    sedangAktifBergerak = (magnitudoAksel > AKTIVITAS_THRESHOLD_G);
  } else {
    // MODE AMAN: jika IMU tidak terbaca, anggap pasien DIAM
    // (asumsi konservatif) supaya alarm keras (buzzer) tetap
    // bisa dipicu -- lebih aman daripada mengasumsikan "aktif"
    // yang hanya memicu getaran senyap.
    sedangAktifBergerak = false;
  }

  // 4. Evaluasi status kondisi (placeholder inferensi)
  statusKandidat = runInferensiStatus(gsrTerfilter, variabilitasEKG, suhuCelsius);

  // 5. Hysteresis: status baru harus konsisten beberapa kali
  //    berturut-turut sebelum benar-benar diubah, supaya tidak flicker
  if (statusKandidat == statusSaatIni) {
    hitunganKonsisten = 0;
  } else {
    hitunganKonsisten++;
    if (hitunganKonsisten >= JUMLAH_KONSISTEN_UNTUK_UBAH_STATUS) {
      statusSaatIni = statusKandidat;
      hitunganKonsisten = 0;
    }
  }

  // 6. Eksekusi aksi fisik sesuai status DAN konteks aktivitas
  //    (context-aware, sesuai Bab 4.4 proposal):
  //    - WASPADA & pasien aktif bergerak -> getaran senyap
  //    - WASPADA/KRITIS & pasien diam/tidur -> buzzer keras
  //    - KRITIS selalu memicu buzzer keras terlepas dari status gerak,
  //      karena ini kondisi darurat yang harus menarik perhatian
  digitalWrite(pinBuzzer, LOW);
  digitalWrite(pinMotorGetar, LOW);

  if (statusSaatIni == KRITIS) {
    digitalWrite(pinBuzzer, HIGH);
    // TODO (hardware asli): trigger GPS NEO-6M untuk ambil koordinat,
    // lalu kirim notifikasi darurat via BLE -> aplikasi ponsel -> MQTT/HTTPS
    // ke server cloud & keluarga. Di simulator ini fungsi tersebut
    // tidak tersedia sehingga hanya ditandai di Serial Monitor.
    Serial.println(">>> STATUS KRITIS: alarm keras aktif, notifikasi darurat perlu dikirim (lihat TODO GPS/BLE/MQTT)");
  } else if (statusSaatIni == WASPADA) {
    if (sedangAktifBergerak) {
      digitalWrite(pinMotorGetar, HIGH); // getaran senyap saat aktif
    } else {
      digitalWrite(pinBuzzer, HIGH);     // alarm lebih tegas saat diam/tidur
    }
  }
  // status NORMAL: kedua aktuator tetap LOW (sudah di-reset di atas)

  // 7. Logging ke Serial Monitor
  Serial.print("EKG(raw): "); Serial.print(nilaiEKGmentah);
  Serial.print(" | VarEKG: "); Serial.print(variabilitasEKG, 1);
  Serial.print(" | GSR(filtered): "); Serial.print(gsrTerfilter);
  Serial.print(" | Suhu: "); Serial.print(suhuCelsius, 1); Serial.print(" C");
  Serial.print(" | MPU: "); Serial.print(mpuTerkoneksi ? "OK" : "GAGAL");
  Serial.print(" | Gerak: "); Serial.print(sedangAktifBergerak ? "AKTIF" : "DIAM");
  Serial.print(" | Status: ");
  switch (statusSaatIni) {
    case NORMAL:  Serial.println("NORMAL");  break;
    case WASPADA: Serial.println("WASPADA"); break;
    case KRITIS:  Serial.println("KRITIS");  break;
  }
}
