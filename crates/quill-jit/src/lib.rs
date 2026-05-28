mod dialect;
mod frontend;
mod fusion;
mod mlir;
mod options;

pub use dialect::{QuillDialectModule, QuillDialectOp, QuillDialectSink, QuillDialectSource};
pub use frontend::{CompiledPipeline, FrontendAdapter};
pub use fusion::{
    FusionConstraint, FusionLoweringKind, FusionMatch, FusionPattern, FusionRegistry,
    PipelineLowering,
};
pub use mlir::{
    CompiledI64Filter, CompiledPlainSum, CompiledRecordPipeline, FixedColumnInput, MlirBackend,
    MlirColumn, MlirModule, RecordPipelineOutput,
};
pub use options::JitOptions;
pub use quill_plan::{
    AggregateFunc, GroupAggregate, JitBinaryOp, JitError, JitExpr, JitProjection, JitResult,
    JitScalar, JitType, OperatorKind, OperatorProperties, OutputMode, PipelineGraph, PipelineKind,
    PipelineSink, PipelineSource, PipelineStage,
};
pub use quill_runtime::{
    CompiledKernel, FilterProjectKernel, FilterSumKernel, FilterSumValue, FixedColumn,
    KernelBackend, KernelKind, PipelineSpec,
};

pub use mlir::{execute_filter_project, execute_filter_sum};
