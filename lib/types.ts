export type Tier = "full" | "probation" | "rejected";

export type Capability = {
  backend: string;
  device_name: string;
  total_memory_mb: number;
  gflops: number;
  mem_bandwidth_gbps: number;
  host: string;
  platform: string;
  torch_version: string;
  supports_training: boolean;
};

export type MeshNode = {
  node_id: string;
  display_name: string;
  gpu: string;
  backend: string;
  gpu_memory_mb: number;
  max_batch_size: number;
  supports_training: boolean;
  capability: Partial<Capability>;
  owner: string | null;
  agent_version: string;
  joined_at: number;
  last_seen: number;
  active: boolean;
  load: number;
  latency_ms: number;
  allocated_memory_mb: number;
  active_batches: number;
  completed_rounds: number;
  failed_rounds: number;
  samples_trained: number;
  seconds_trained: number;
  reliability: number;
  throughput_sps: number;
  training_epoch: number;
  training_total_epochs: number;
  fitness: number;
  tier?: Tier;
  admission_reason?: string;
};

export type ShardAssignment = {
  node_id: string;
  shard_index: number;
  samples: number;
  fitness: number;
  tier: Tier;
  throughput_sps: number;
  predicted_seconds: number;
};

export type RoundPlan = {
  total_samples: number;
  assignments: ShardAssignment[];
  rejected: { node_id: string; reason: string; fitness: number }[];
  predicted_makespan_seconds: number;
  predicted_serial_seconds: number;
  predicted_speedup: number;
};

export type RoundRecord = {
  round: number;
  workers: number;
  samples: number;
  dropped_shards: number;
  makespan_seconds: number;
  fastest_seconds: number;
  straggler_gap_seconds: number;
  aggregation_seconds: number;
  wall_clock_seconds: number;
  serial_estimate_seconds: number;
  speedup: number;
  efficiency: number;
  shards: {
    node_id: string;
    node_name: string | null;
    samples: number;
    seconds: number;
    predicted_seconds: number;
    tier: Tier;
    weight: number;
    metrics: Record<string, unknown> | null;
  }[];
};

export type RunStatus = "planning" | "waiting" | "running" | "done" | "failed" | "stopped";

export type RunSummary = {
  id: string;
  name: string;
  status: RunStatus;
  mode: "mesh" | "solo";
  created_at: number;
  finished_at: number | null;
  dataset_id: string;
  dataset_name: string;
  base_model: string;
  rounds: number;
  current_round: number;
  imgsz: number;
  batch_size: number;
  total_samples: number;
  wall_clock_seconds: number;
  serial_estimate_seconds: number;
  speedup: number;
  efficiency: number;
  peak_workers: number;
  has_artifact: boolean;
};

export type LiveShard = {
  batch_id: string;
  node_id: string;
  node_name: string | null;
  status: "queued" | "assigned" | "done" | "failed" | "dropped";
  samples: number;
  tier: Tier;
  round: number;
  predicted_seconds: number;
  batch_size: number;
  elapsed_seconds: number | null;
  soft_deadline_seconds: number;
  hard_deadline_seconds: number;
  error: string | null;
};

export type RunDetail = RunSummary & {
  round_history: RoundRecord[];
  plan: RoundPlan | null;
  error: string | null;
  notes: string | null;
  class_names: string[];
  live_shards: LiveShard[];
};

export type Dataset = {
  id: string;
  name: string;
  filename: string;
  created_at: number;
  bytes: number;
  train_count: number;
  val_count: number;
  class_names: string[];
  is_default?: boolean;
};

export type MeshPolicy = Record<string, number>;

export type MeshState = {
  mesh_name: string;
  nodes: MeshNode[];
  metrics: {
    nodes_total: number;
    nodes_active: number;
    nodes_training: number;
    nodes_admitted: number;
    total_gflops: number;
    total_memory_mb: number;
    active_shards: number;
    stream_subscribers: number;
  };
  active_runs: RunSummary[];
  plan_preview: RoundPlan;
  policy: MeshPolicy;
  torch_ready: boolean;
  default_dataset: Dataset | null;
};

export type MeshEvent = {
  id: number;
  kind: string;
  at: number;
  data: Record<string, unknown>;
};

export type NetworkDevice = {
  ip: string;
  mac: string | null;
  hostname: string | null;
  open_ports: number[];
  is_coordinator: boolean;
  is_this_host: boolean;
  is_member: boolean;
  is_visitor: boolean;
  source: "arp" | "scan";
  /** Measured TCP round trip. Null when the device answers on no port. */
  rtt_ms: number | null;
  proximity: "close" | "nearby" | "far" | "unknown";
};

export type Visitor = {
  visitor_id: string;
  name: string;
  platform: string | null;
  user_agent: string | null;
  cores: number | null;
  memory_gb: number | null;
  gpu: string | null;
  webgpu: boolean;
  screen: string | null;
  ip: string | null;
  first_seen: number;
  last_seen: number;
  has_agent: boolean;
};

export type DiscoverState = {
  mdns: {
    active: boolean;
    hostname: string;
    address: string | null;
    web_port: number;
    api_port: number;
    error: string | null;
  };
  addresses: { lan_ip: string; web_port: number; api_port: number };
  scan: {
    subnet: string | null;
    at: number;
    running: boolean;
    duration_seconds: number;
    scanned: number;
  };
  devices: NetworkDevice[];
  visitors: Visitor[];
  nodes: {
    node_id: string;
    display_name: string | null;
    gpu: string | null;
    backend: string | null;
    active: boolean;
    host: string | null;
  }[];
  idle_count: number;
};

// ---------------------------------------------------------------------------
// Benchmark suites
// ---------------------------------------------------------------------------

export type PartitionStrategy = "proportional" | "equal";

export type SuiteConfig = {
  name: string;
  base_model: string;
  parent_dataset_id: string | null;
  dataset_sizes: number[];
  node_counts: number[];
  strategies: PartitionStrategy[];
  repeats: number;
  rounds: number;
  imgsz: number;
  batch_size: number;
  node_selection: "strongest" | "random";
  evaluate: boolean;
  network_label: string;
  notes: string;
  trial_timeout_seconds: number;
  settle_seconds: number;
};

export type TrialSpec = {
  trial_id: string;
  index: number;
  node_count: number;
  sample_count: number;
  strategy: PartitionStrategy;
  repeat: number;
  node_ids: string[];
  dataset_id: string | null;
  label: string;
};

export type TrialResult = {
  trial_id: string;
  index: number;
  label: string;
  status: "done" | "failed" | "skipped" | "aborted";
  error: string | null;
  run_id: string | null;
  node_count: number;
  sample_count: number;
  strategy: PartitionStrategy;
  repeat: number;
  rounds_completed: number;
  train_seconds: number;
  serial_estimate_seconds: number;
  eval_seconds: number;
  /** Against the one-machine baseline at the same dataset size. */
  speedup: number | null;
  efficiency: number | null;
  /** Shard-time overlap inside a round. Diagnostic, not the headline. */
  parallel_speedup?: number | null;
  baseline_train_seconds?: number | null;
  map50: number | null;
  map50_95: number | null;
  mean_imbalance: number | null;
  mean_straggler_gap_seconds: number | null;
  comm_bytes: number;
  comm_seconds: number;
  comm_fraction: number | null;
};

export type SuiteCell = {
  node_count: number;
  sample_count: number;
  strategy: PartitionStrategy;
  runs: number;
  train_seconds_mean: number | null;
  train_seconds_std: number | null;
  speedup_mean: number | null;
  speedup_std: number | null;
  parallel_speedup_mean?: number | null;
  efficiency_mean: number | null;
  efficiency_std: number | null;
  map50_mean: number | null;
  map50_std: number | null;
  mean_imbalance_mean: number | null;
  baseline_map50: number | null;
  delta_map50: number | null;
};

export type Suite = {
  id: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  status: "pending" | "running" | "done" | "aborted" | "failed" | "interrupted";
  config: SuiteConfig;
  trials: TrialSpec[];
  results: TrialResult[];
  current_trial: TrialSpec | null;
  cells: SuiteCell[];
  estimated_seconds: number;
  is_active?: boolean;
  error?: string | null;
  environment: {
    coordinator: Record<string, unknown>;
    dataset: { id: string; name: string; train_count: number; val_count: number; class_names: string[] };
    nodes: {
      node_id: string;
      name: string | null;
      gpu: string | null;
      backend: string | null;
      memory_mb: number | null;
      capability: Record<string, unknown> | null;
    }[];
  };
};

export type SuiteIndexEntry = {
  id: string;
  name: string;
  status: Suite["status"];
  created_at: number;
  finished_at: number | null;
  total_trials: number;
  completed_trials: number;
  failed_trials: number;
  network_label: string;
};

export type BenchmarkIndex = {
  suites: SuiteIndexEntry[];
  active_suite_id: string | null;
  available_nodes: number;
  nodes: {
    node_id: string;
    name: string | null;
    gpu: string | null;
    backend: string | null;
    memory_mb: number | null;
    gflops: number | null;
    throughput_sps: number | null;
  }[];
  torch_ready: boolean;
};

export type SuitePreview = {
  trials: number;
  estimated_seconds: number;
  available_nodes: number;
  breakdown: TrialSpec[];
};
