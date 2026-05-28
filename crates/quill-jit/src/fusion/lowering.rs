use quill_plan::{
    JitExpr, JitProjection, PipelineGraph, PipelineKind, PipelineSink, PipelineStage,
};

use super::registry::FusionRegistry;
use super::FusionLoweringKind;

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineLowering {
    Record {
        predicate: JitExpr,
        projections: Vec<JitProjection>,
    },
    PlainSum {
        predicate: JitExpr,
        measure: JitExpr,
    },
}

impl PipelineLowering {
    pub fn from_graph(graph: &PipelineGraph) -> Option<Self> {
        FusionRegistry::builtin()
            .match_pipeline(graph)
            .map(|matched| matched.lowering)
    }

    pub fn kind(&self) -> PipelineKind {
        match self {
            Self::Record { .. } => PipelineKind::Record,
            Self::PlainSum { .. } => PipelineKind::Aggregate,
        }
    }
}

pub(crate) fn extract_lowering(
    kind: FusionLoweringKind,
    graph: &PipelineGraph,
) -> Option<PipelineLowering> {
    match kind {
        FusionLoweringKind::Record => extract_record_lowering(graph),
        FusionLoweringKind::PlainSum => extract_plain_sum_lowering(graph),
    }
}

fn extract_record_lowering(graph: &PipelineGraph) -> Option<PipelineLowering> {
    match graph.stages.as_slice() {
        [PipelineStage::Filter(predicate), PipelineStage::Projection(projections)] => {
            Some(PipelineLowering::Record {
                predicate: predicate.clone(),
                projections: projections.clone(),
            })
        }
        _ => None,
    }
}

fn extract_plain_sum_lowering(graph: &PipelineGraph) -> Option<PipelineLowering> {
    let [PipelineStage::Filter(predicate)] = graph.stages.as_slice() else {
        return None;
    };
    let PipelineSink::Sum { measure } = &graph.sink else {
        return None;
    };
    Some(PipelineLowering::PlainSum {
        predicate: predicate.clone(),
        measure: measure.clone(),
    })
}
