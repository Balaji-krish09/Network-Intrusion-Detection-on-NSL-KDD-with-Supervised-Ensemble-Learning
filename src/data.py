"""Loading NSL-KDD and mapping attack names to the 4 standard categories."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
BINARY = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Covers every attack name in KDDTrain+ and KDDTest+ (the test set adds 17 new ones).
ATTACK_CATEGORY = {
    "normal": "Normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS", "worm": "DoS",
    # Probing
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load(name: str) -> pd.DataFrame:
    """Load one split, e.g. 'KDDTrain+', 'KDDTest+', 'KDDTest-21', 'KDDTrain+_20Percent'."""
    df = pd.read_csv(DATA_DIR / f"{name}.txt", header=None, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY)
    missing = df.loc[df["category"].isna(), "label"].unique()
    if len(missing):
        raise ValueError(f"Unmapped attack labels: {missing}")
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def load_all():
    return load("KDDTrain+"), load("KDDTest+"), load("KDDTest-21")
