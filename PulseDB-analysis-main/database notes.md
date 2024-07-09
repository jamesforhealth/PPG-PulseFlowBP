# sqlite database
## notes
data_segment.array_index: corresponds to PulseDB segment index from MATLAB.  Ex. 3 -> 3rd segment of data

patient_info_snapshot.identifier: ex. "p005050"

patient_info_snapshot properties (weight, height, age) are as of when the measurement was taken.  patient_info_snapshot.id does not represent a unique patient.

patient.id represents a unique patient.  patient.GUID could be: national ID number, passport number, etc.

## schema design strategy
- want table relations to support filtering operations
- want idempotent key-value store
- patient_info_snapshot keys: data_source, identifier; values: gender, weight_kg, height_cm, age
- data_segment keys: data_source, patient_snapshot_id, array_index; values: sample_rate_hz
- analysis tables keys: segment_id
