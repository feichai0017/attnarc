mod dialect;
mod frontend;
mod fusion;
mod mlir;
mod options;
mod spec;

pub use dialect::{QuillDialectModule, QuillDialectOp, QuillDialectSink, QuillDialectSource};
pub use frontend::{CompiledPipeline, FrontendAdapter};
pub use fusion::{
    FusionConstraint, FusionLoweringKind, FusionMatch, FusionPattern, FusionRegistry,
    PipelineLowering,
};
pub use mlir::{
    CompiledGroupAggregateUpdate, CompiledPlainSum, CompiledRecordPipeline, FixedColumnInput,
    MlirBackend, MlirColumn, MlirModule, RecordPipelineOutput,
};
pub use options::JitOptions;
pub use quill_plan::{
    AggregateFunc, GroupAggregate, JitBinaryOp, JitError, JitExpr, JitProjection, JitResult,
    JitScalar, JitType, OperatorKind, OperatorProperties, OutputMode, PipelineGraph, PipelineKind,
    PipelineSink, PipelineSource, PipelineStage,
};
pub use quill_runtime::{FilterProjectKernel, FilterSumKernel, FilterSumValue};
pub use spec::{CompiledKernel, FixedColumn, KernelKind, PipelineSpec};

pub use mlir::{execute_filter_project, execute_filter_sum, execute_group_aggregate_update};
