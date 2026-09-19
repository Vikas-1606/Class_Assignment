"""Data ingestion, synthesis, and parsing module for System Log Anomaly Detection.

Implements high-throughput parsing of unstructured system logs into structured
event templates and session-grouped sequential telemetry adhering to DeepLog
standards (Du et al., ACM CCS 2017).
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is on sys.path for direct script execution
PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

import click
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_LOG_FILE,
    LABEL_FILE,
    PARSED_EVENTS_FILE,
    SESSION_SEQUENCES_FILE,
    TEMPLATES_FILE,
    RANDOM_SEED,
    ensure_directories,
)

# Canonical benchmark log templates derived from HDFS and distributed systems
HDFS_TEMPLATES: List[Dict[str, str]] = [
    {
        "EventId": "E1",
        "Template": "Receiving block <*> src: <*> dest: <*>",
        "Regex": r"Receiving block (blk_-?\d+) src: (\S+) dest: (\S+)",
        "Component": "dfs.DataNode$DataXceiver",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E2",
        "Template": "BLOCK* NameSystem.allocateBlock: <*> <*>",
        "Regex": r"BLOCK\* NameSystem\.allocateBlock: (\S+) (blk_-?\d+)",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E3",
        "Template": "PacketResponder <*> for block <*> terminating",
        "Regex": r"PacketResponder (\d+) for block (blk_-?\d+) terminating",
        "Component": "dfs.DataNode$PacketResponder",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E4",
        "Template": "Received block <*> of size <*> from <*>",
        "Regex": r"Received block (blk_-?\d+) of size (\d+) from (\S+)",
        "Component": "dfs.DataNode$PacketResponder",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E5",
        "Template": "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>",
        "Regex": r"BLOCK\* NameSystem\.addStoredBlock: blockMap updated: (\S+) is added to (blk_-?\d+) size (\d+)",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E6",
        "Template": "<*> Served block <*> to <*>",
        "Regex": r"(\S+) Served block (blk_-?\d+) to (\S+)",
        "Component": "dfs.DataNode$DataXceiver",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E7",
        "Template": "Verification succeeded for block <*>",
        "Regex": r"Verification succeeded for block (blk_-?\d+)",
        "Component": "dfs.DataNode$BlockReceiver",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E8",
        "Template": "Starting thread to transfer block <*> to <*>",
        "Regex": r"Starting thread to transfer block (blk_-?\d+) to (\S+)",
        "Component": "dfs.DataNode$DataTransfer",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E9",
        "Template": "BLOCK* ask <*> to replicate <*> to datanode(s) <*>",
        "Regex": r"BLOCK\* ask (\S+) to replicate (blk_-?\d+) to datanode\(s\) (\S+)",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E10",
        "Template": "BLOCK* ask <*> to delete <*>",
        "Regex": r"BLOCK\* ask (\S+) to delete (blk_-?\d+)",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E11",
        "Template": "Deleting block <*> file <*>",
        "Regex": r"Deleting block (blk_-?\d+) file (\S+)",
        "Component": "dfs.DataNode$FSDataset",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E12",
        "Template": "BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>",
        "Regex": r"BLOCK\* NameSystem\.delete: (blk_-?\d+) is added to invalidSet of (\S+)",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E13",
        "Template": "Transmitted block <*> to <*>",
        "Regex": r"Transmitted block (blk_-?\d+) to (\S+)",
        "Component": "dfs.DataNode$DataTransfer",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E14",
        "Template": "Updating block <*> to state <*>",
        "Regex": r"Updating block (blk_-?\d+) to state (\S+)",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    {
        "EventId": "E15",
        "Template": "BLOCK* removing block <*> from neededReplications as it does not belong to any file",
        "Regex": r"BLOCK\* removing block (blk_-?\d+) from neededReplications as it does not belong to any file",
        "Component": "dfs.FSNamesystem",
        "Level": "INFO",
        "Type": "normal",
    },
    # Anomalous operational event templates
    {
        "EventId": "E16",
        "Template": "PacketResponder <*> for block <*> Interrupted",
        "Regex": r"PacketResponder (\d+) for block (blk_-?\d+) Interrupted",
        "Component": "dfs.DataNode$PacketResponder",
        "Level": "WARN",
        "Type": "anomaly",
    },
    {
        "EventId": "E17",
        "Template": "Unexpected error trying to delete block <*>. BlockInfo not found",
        "Regex": r"Unexpected error trying to delete block (blk_-?\d+)\. BlockInfo not found",
        "Component": "dfs.FSNamesystem",
        "Level": "ERROR",
        "Type": "anomaly",
    },
    {
        "EventId": "E18",
        "Template": "Verification failed for block <*>",
        "Regex": r"Verification failed for block (blk_-?\d+)",
        "Component": "dfs.DataNode$BlockReceiver",
        "Level": "ERROR",
        "Type": "anomaly",
    },
    {
        "EventId": "E19",
        "Template": "Exception in receiveBlock for block <*>: Connection reset by peer",
        "Regex": r"Exception in receiveBlock for block (blk_-?\d+): Connection reset by peer",
        "Component": "dfs.DataNode$BlockReceiver",
        "Level": "ERROR",
        "Type": "anomaly",
    },
    {
        "EventId": "E20",
        "Template": "Write block <*> failed with timeout after <*> ms",
        "Regex": r"Write block (blk_-?\d+) failed with timeout after (\d+) ms",
        "Component": "dfs.DataNode$DataXceiver",
        "Level": "ERROR",
        "Type": "anomaly",
    },
    {
        "EventId": "E21",
        "Template": "BLOCK* Cannot replicate block <*>, no target datanode available",
        "Regex": r"BLOCK\* Cannot replicate block (blk_-?\d+), no target datanode available",
        "Component": "dfs.FSNamesystem",
        "Level": "WARN",
        "Type": "anomaly",
    },
    {
        "EventId": "E22",
        "Template": "PendingReplicationMonitor timed out block <*>",
        "Regex": r"PendingReplicationMonitor timed out block (blk_-?\d+)",
        "Component": "dfs.FSNamesystem",
        "Level": "WARN",
        "Type": "anomaly",
    },
    {
        "EventId": "E23",
        "Template": "DataStreamer Exception: block <*> generation stamp mismatch",
        "Regex": r"DataStreamer Exception: block (blk_-?\d+) generation stamp mismatch",
        "Component": "dfs.DataStreamer",
        "Level": "ERROR",
        "Type": "anomaly",
    },
    {
        "EventId": "E24",
        "Template": "Checksum error while reading block <*> at offset <*>",
        "Regex": r"Checksum error while reading block (blk_-?\d+) at offset (\d+)",
        "Component": "dfs.DataNode$BlockReceiver",
        "Level": "ERROR",
        "Type": "anomaly",
    },
    {
        "EventId": "E25",
        "Template": "Target datanode <*> unreachable for block <*>",
        "Regex": r"Target datanode (\S+) unreachable for block (blk_-?\d+)",
        "Component": "dfs.DataNode$DataTransfer",
        "Level": "WARN",
        "Type": "anomaly",
    },
]

# Normal sequential lifecycle grammar pathways
NORMAL_PATTERNS = [
    # Path A: Standard write, store, verify, serve
    ["E2", "E1", "E3", "E4", "E5", "E5", "E5", "E7", "E6"],
    # Path B: Standard write, replicate, serve
    ["E2", "E1", "E3", "E4", "E5", "E8", "E13", "E5", "E7", "E6", "E6"],
    # Path C: Write, store, delete
    ["E2", "E1", "E3", "E4", "E5", "E5", "E7", "E12", "E10", "E11"],
    # Path D: Multi-replica write, state update, replication monitor
    ["E2", "E1", "E1", "E3", "E3", "E4", "E5", "E5", "E5", "E14", "E9", "E8", "E13", "E7", "E6"],
    # Path E: Clean write and read cycle
    ["E2", "E1", "E3", "E4", "E5", "E5", "E7", "E6", "E6", "E6"],
]

# Anomaly patterns injecting subtle, out-of-order, or catastrophic event faults
ANOMALY_PATTERNS = [
    # Anomaly 1: Connection reset during block receive
    ["E2", "E1", "E19", "E20"],
    # Anomaly 2: Packet responder interruption
    ["E2", "E1", "E16", "E20", "E12"],
    # Anomaly 3: Verification checksum failure
    ["E2", "E1", "E3", "E4", "E5", "E18", "E24"],
    # Anomaly 4: Replication timeout and missing target node
    ["E2", "E1", "E3", "E4", "E5", "E9", "E21", "E22"],
    # Anomaly 5: Generation stamp mismatch and unreachable node
    ["E2", "E1", "E3", "E4", "E23", "E25", "E17"],
    # Anomaly 6: Sequence disorder (delete called before block is even written or stored)
    ["E2", "E10", "E11", "E17", "E1"],
]


def generate_synthetic_benchmark_dataset(
    num_sessions: int = 4000,
    anomaly_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Generate a realistic, large-scale benchmark HDFS log dataset.

    Args:
        num_sessions: Total number of block sessions to generate.
        anomaly_ratio: Fraction of sessions that exhibit anomalous behaviors.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (raw_log_lines, session_labels).
    """
    random.seed(seed)
    raw_log_lines: List[str] = []
    session_labels: List[Dict[str, str]] = []

    ip_pool = [
        "10.250.19.102",
        "10.250.14.224",
        "10.251.43.115",
        "10.251.71.18",
        "10.250.10.6",
        "10.251.111.204",
    ]
    template_map = {t["EventId"]: t for t in HDFS_TEMPLATES}

    num_anomalous = int(num_sessions * anomaly_ratio)
    num_normal = num_sessions - num_anomalous

    session_types = ["Normal"] * num_normal + ["Anomaly"] * num_anomalous
    random.shuffle(session_types)

    timestamp_base = 1226250000  # Unix timestamp base
    pid = 143

    for i, s_type in enumerate(session_types):
        block_num = 1000000000 + i
        block_id = f"blk_{block_num}"
        session_labels.append({"BlockId": block_id, "Label": s_type})

        if s_type == "Normal":
            base_pattern = list(random.choice(NORMAL_PATTERNS))
            # Occasionally inject small natural repetitions (e.g. repeated E6 served block)
            if random.random() < 0.3:
                base_pattern.extend(["E6"] * random.randint(1, 3))
            events = base_pattern
        else:
            events = list(random.choice(ANOMALY_PATTERNS))
            # Inject some normal prefix before anomaly
            if random.random() < 0.4:
                prefix = random.choice([["E2"], ["E2", "E1"], ["E2", "E1", "E3"]])
                events = prefix + events

        src_ip = random.choice(ip_pool)
        dest_ip = random.choice(ip_pool)
        size = random.choice([67108864, 134217728, 33554432, 268435456])

        for event_id in events:
            timestamp_base += random.randint(1, 4)
            date_str = "081109"
            time_str = f"{timestamp_base % 86400 // 3600:02d}{(timestamp_base % 3600) // 60:02d}{timestamp_base % 60:02d}"
            tmpl = template_map[event_id]

            # Materialize event template into concrete log message
            msg = tmpl["Template"]
            if event_id == "E1":
                msg = f"Receiving block {block_id} src: /{src_ip}:54106 dest: /{dest_ip}:50010"
            elif event_id == "E2":
                msg = f"BLOCK* NameSystem.allocateBlock: /user/root/data/{block_id}.txt {block_id}"
            elif event_id == "E3":
                msg = f"PacketResponder 1 for block {block_id} terminating"
            elif event_id == "E4":
                msg = f"Received block {block_id} of size {size} from /{src_ip}"
            elif event_id == "E5":
                node_ip = random.choice(ip_pool)
                msg = f"BLOCK* NameSystem.addStoredBlock: blockMap updated: {node_ip}:50010 is added to {block_id} size {size}"
            elif event_id == "E6":
                node_ip = random.choice(ip_pool)
                target_ip = random.choice(ip_pool)
                msg = f"{node_ip}:50010 Served block {block_id} to /{target_ip}"
            elif event_id == "E7":
                msg = f"Verification succeeded for block {block_id}"
            elif event_id == "E8":
                target_ip = random.choice(ip_pool)
                msg = f"Starting thread to transfer block {block_id} to {target_ip}:50010"
            elif event_id == "E9":
                msg = f"BLOCK* ask {src_ip}:50010 to replicate {block_id} to datanode(s) {dest_ip}:50010"
            elif event_id == "E10":
                msg = f"BLOCK* ask {src_ip}:50010 to delete {block_id}"
            elif event_id == "E11":
                msg = f"Deleting block {block_id} file /mnt/hadoop/dfs/data/{block_id}"
            elif event_id == "E12":
                msg = f"BLOCK* NameSystem.delete: {block_id} is added to invalidSet of {src_ip}:50010"
            elif event_id == "E13":
                msg = f"Transmitted block {block_id} to /{dest_ip}:50010"
            elif event_id == "E14":
                msg = f"Updating block {block_id} to state COMMITTED"
            elif event_id == "E15":
                msg = f"BLOCK* removing block {block_id} from neededReplications as it does not belong to any file"
            elif event_id == "E16":
                msg = f"PacketResponder 2 for block {block_id} Interrupted"
            elif event_id == "E17":
                msg = f"Unexpected error trying to delete block {block_id}. BlockInfo not found"
            elif event_id == "E18":
                msg = f"Verification failed for block {block_id}"
            elif event_id == "E19":
                msg = f"Exception in receiveBlock for block {block_id}: Connection reset by peer"
            elif event_id == "E20":
                timeout = random.randint(30000, 60000)
                msg = f"Write block {block_id} failed with timeout after {timeout} ms"
            elif event_id == "E21":
                msg = f"BLOCK* Cannot replicate block {block_id}, no target datanode available"
            elif event_id == "E22":
                msg = f"PendingReplicationMonitor timed out block {block_id}"
            elif event_id == "E23":
                msg = f"DataStreamer Exception: block {block_id} generation stamp mismatch"
            elif event_id == "E24":
                msg = f"Checksum error while reading block {block_id} at offset 4096"
            elif event_id == "E25":
                msg = f"Target datanode {dest_ip}:50010 unreachable for block {block_id}"

            log_line = f"{date_str} {time_str} {pid} {tmpl['Level']} {tmpl['Component']}: {msg}"
            raw_log_lines.append(log_line)

    return raw_log_lines, session_labels


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single raw log string into structured components and event template.

    Format: Date Time Pid Level Component: Message
    """
    header_pattern = r"^(\d{6})\s+(\d{6})\s+(\d+)\s+([A-Z]+)\s+([\w\.\$]+):\s+(.*)$"
    match = re.match(header_pattern, line.strip())
    if not match:
        return None

    date, time, pid, level, component, content = match.groups()

    # Extract BlockId
    block_match = re.search(r"(blk_-?\d+)", content)
    block_id = block_match.group(1) if block_match else "UNKNOWN_BLOCK"

    # Match against compiled regex templates
    matched_event_id = "E_UNKNOWN"
    matched_template = content

    for tmpl in HDFS_TEMPLATES:
        if re.search(tmpl["Regex"], content):
            matched_event_id = tmpl["EventId"]
            matched_template = tmpl["Template"]
            break

    return {
        "Date": date,
        "Time": time,
        "Pid": int(pid),
        "Level": level,
        "Component": component,
        "Content": content,
        "BlockId": block_id,
        "EventId": matched_event_id,
        "EventTemplate": matched_template,
    }


def parse_raw_logs(
    raw_log_path: Path,
    label_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Parse raw unstructured log file and group into sequential telemetry sessions.

    Args:
        raw_log_path: Path to the raw log file.
        label_path: Path to the CSV with columns [BlockId, Label].

    Returns:
        A tuple of (parsed_events_df, session_dict).
    """
    records: List[Dict[str, Any]] = []

    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                records.append(parsed)

    df_events = pd.DataFrame(records)

    # Load session anomaly labels if present
    labels_map: Dict[str, str] = {}
    if label_path.exists():
        df_labels = pd.read_csv(label_path)
        labels_map = dict(zip(df_labels["BlockId"], df_labels["Label"]))

    # Group by BlockId to reconstruct execution sequence
    sessions: Dict[str, Dict[str, Any]] = {}
    for block_id, group in df_events.groupby("BlockId"):
        if block_id == "UNKNOWN_BLOCK":
            continue
        event_seq = group["EventId"].tolist()
        label = labels_map.get(block_id, "Normal")
        sessions[block_id] = {
            "events": event_seq,
            "label": label,
            "length": len(event_seq),
        }

    return df_events, sessions


def persist_interim_data(
    df_events: pd.DataFrame,
    sessions: Dict[str, Dict[str, Any]],
    interim_dir: Path = INTERIM_DATA_DIR,
) -> None:
    """Save parsed log events, session sequences, and template mapping to interim dir."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    df_events.to_csv(PARSED_EVENTS_FILE, index=False)

    with open(SESSION_SEQUENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

    with open(TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump(HDFS_TEMPLATES, f, indent=2)


@click.command()
@click.option(
    "--download-or-generate",
    is_flag=True,
    default=True,
    help="Download or generate synthetic benchmark HDFS dataset if not found.",
)
@click.option(
    "--num-sessions",
    default=4000,
    type=int,
    help="Number of sessions to generate if synthesizing dataset.",
)
@click.option(
    "--anomaly-ratio",
    default=0.15,
    type=float,
    help="Fraction of anomalous sessions in generated dataset.",
)
@click.option(
    "--raw-path",
    type=click.Path(),
    default=None,
    help="Path to an existing raw log file.",
)
def main(
    download_or_generate: bool,
    num_sessions: int,
    anomaly_ratio: float,
    raw_path: Optional[str],
) -> None:
    """CLI Entrypoint for Dataset Preparation."""
    ensure_directories()
    click.echo("=" * 70)
    click.echo("Step 1: System Telemetry Log Ingestion & Parsing")
    click.echo("=" * 70)

    target_raw_log = Path(raw_path) if raw_path else RAW_LOG_FILE

    # If raw log does not exist and flag is set, generate benchmark data
    if not target_raw_log.exists():
        if download_or_generate:
            click.echo(f"Raw log not found at {target_raw_log}. Generating benchmark dataset...")
            raw_lines, labels = generate_synthetic_benchmark_dataset(
                num_sessions=num_sessions,
                anomaly_ratio=anomaly_ratio,
            )
            with open(RAW_LOG_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(raw_lines) + "\n")

            df_lbl = pd.DataFrame(labels)
            df_lbl.to_csv(LABEL_FILE, index=False)
            click.echo(f"Saved {len(raw_lines)} raw log records to {RAW_LOG_FILE}")
            click.echo(f"Saved {len(labels)} session labels to {LABEL_FILE}")
        else:
            raise FileNotFoundError(
                f"Raw log file not found at {target_raw_log}. Use --download-or-generate to create it."
            )

    click.echo(f"Parsing raw log telemetry from: {target_raw_log}...")
    df_events, sessions = parse_raw_logs(target_raw_log, LABEL_FILE)

    click.echo(f"Parsed {len(df_events)} events across {len(sessions)} distinct block sessions.")
    persist_interim_data(df_events, sessions)

    normal_count = sum(1 for s in sessions.values() if s["label"] == "Normal")
    anomaly_count = sum(1 for s in sessions.values() if s["label"] == "Anomaly")
    click.echo(f"Session distribution: Normal = {normal_count}, Anomaly = {anomaly_count}")
    click.echo(f"Parsed events saved to: {PARSED_EVENTS_FILE}")
    click.echo(f"Session sequences saved to: {SESSION_SEQUENCES_FILE}")
    click.echo("Dataset ingestion and parsing completed successfully.\n")


if __name__ == "__main__":
    main()
