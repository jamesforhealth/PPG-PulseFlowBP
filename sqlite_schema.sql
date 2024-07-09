
CREATE TABLE "patient" (
	"id"	INTEGER NOT NULL UNIQUE,
	"GUID"	TEXT NOT NULL UNIQUE,
	"birthday_yyyymmdd"	TEXT,
	"phone_number"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE "data_source" (
	"id"	INTEGER NOT NULL UNIQUE,
	"name"	TEXT NOT NULL UNIQUE,
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE "gender" (
	"id"	INTEGER NOT NULL UNIQUE,
	"gender"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE "patient_info_snapshot" (
	"id"	INTEGER NOT NULL UNIQUE,
	"patient_id"	INTEGER,
	"data_source"	INTEGER NOT NULL,
	"identifier"	TEXT NOT NULL,
	"gender"	INTEGER,
	"weight_kg"	REAL,
	"height_cm"	REAL,
	"age"	INTEGER,
	PRIMARY KEY("id"),
	FOREIGN KEY("gender") REFERENCES "gender"("id"),
	FOREIGN KEY("patient_id") REFERENCES "patient"("id"),
	FOREIGN KEY("data_source") REFERENCES "data_source"("id")
);

CREATE TABLE "data_segment" (
	"id"	INTEGER NOT NULL UNIQUE,
	"data_source"	INTEGER NOT NULL,
	"patient_snapshot_id"	INTEGER NOT NULL,
	"array_index"	INTEGER NOT NULL,
	"sample_rate_hz"	REAL NOT NULL,
	FOREIGN KEY("patient_snapshot_id") REFERENCES "patient_info_snapshot"("id"),
	PRIMARY KEY("id" AUTOINCREMENT)
);


CREATE TABLE "basic_analysis" (
	"id"	INTEGER NOT NULL UNIQUE,
	"segment_id"	INTEGER NOT NULL,
	"heart_rate_bpm"	REAL,
	"mean_arterial_pressure_mmhg"	REAL,
	"pulse_pressure_mmhg"	REAL,
	"systolic_pressure_mmhg"	REAL,
	"diastolic_pressure_mmhg"	REAL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("segment_id") REFERENCES "data_segment"("id")
);

CREATE TABLE "normalized_wk3_analysis" (
	"id"	INTEGER NOT NULL UNIQUE,
	"segment_id"	INTEGER NOT NULL,
	"R"	REAL,
	"C"	REAL,
	"Zc"	REAL,
	"root_mean_squared_error"	REAL,
	"root_mean_squared_error_systole"	REAL,
	"root_mean_squared_error_diastole"	REAL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("segment_id") REFERENCES "data_segment"("id")
);

CREATE TABLE "wk3_analysis" (
	"id"	INTEGER NOT NULL UNIQUE,
	"segment_id"	INTEGER NOT NULL,
	"R"	REAL,
	"C"	REAL,
	"Zc"	REAL,
	"root_mean_squared_error"	REAL,
	"root_mean_squared_error_systole"	REAL,
	"root_mean_squared_error_diastole"	REAL,
	FOREIGN KEY("segment_id") REFERENCES "data_segment"("id"),
	PRIMARY KEY("id" AUTOINCREMENT)
);

INSERT INTO data_source ("name") VALUES ("PulseDB - Vital");
INSERT INTO data_source ("name") VALUES ("PulseDB - MIMIC");
INSERT INTO gender ("gender") VALUES ("male");
INSERT INTO gender ("gender") VALUES ("female");