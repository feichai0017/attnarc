#ifndef QUILL_IR_QUILLOPS_TD
#define QUILL_IR_QUILLOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Quill_Dialect : Dialect {
  let name = "quill";
  let summary = "QuillSQL pipeline dialect";
  let description = [{
    The Quill dialect models vectorized SQL pipelines between frontend
    physical plans and low-level MLIR loop lowering.  It keeps row, batch,
    selection, and sink semantics explicit while reusing arith for scalar
    expression trees.
  }];
  let cppNamespace = "::mlir::quill";
  let useDefaultTypePrinterParser = 1;
}

class Quill_Type<string name, string typeMnemonic> : TypeDef<Quill_Dialect, name> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "";
}

def Quill_BatchType : Quill_Type<"Batch", "batch">;
def Quill_SelectionType : Quill_Type<"Selection", "selection">;
def Quill_GroupIdsType : Quill_Type<"GroupIds", "group_ids">;
def Quill_RowType : Quill_Type<"Row", "row">;
def Quill_ScalarType : Quill_Type<"Scalar", "scalar">;

def Quill_Batch :
    Type<CPred<"::llvm::isa<::mlir::quill::BatchType>($_self)">,
         "Quill batch">;
def Quill_Selection :
    Type<CPred<"::llvm::isa<::mlir::quill::SelectionType>($_self)">,
         "Quill selection">;
def Quill_GroupIds :
    Type<CPred<"::llvm::isa<::mlir::quill::GroupIdsType>($_self)">,
         "Quill group ids">;
def Quill_Row :
    Type<CPred<"::llvm::isa<::mlir::quill::RowType>($_self)">,
         "Quill row">;
def Quill_Scalar :
    Type<CPred<"::llvm::isa<::mlir::quill::ScalarType>($_self)">,
         "Quill scalar">;

def Quill_AggregateFuncArrayAttr
    : TypedArrayAttrBase<StrAttr, "Quill aggregate function array"> {
  let predicate = And<[
    TypedArrayAttrBase<StrAttr, "Quill aggregate function array">.predicate,
    CPred<[{
      ::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self),
                     [](::mlir::Attribute attr) {
        auto value = ::llvm::cast<::mlir::StringAttr>(attr).getValue();
        return value == "sum" || value == "count" || value == "avg" ||
               value == "min" || value == "max";
      })
    }]>
  ]>;
}

def Quill_AggregateStateTypeArrayAttr
    : TypedArrayAttrBase<StrAttr, "Quill aggregate state type array"> {
  let predicate = And<[
    TypedArrayAttrBase<StrAttr, "Quill aggregate state type array">.predicate,
    CPred<[{
      ::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self),
                     [](::mlir::Attribute attr) {
        auto value = ::llvm::cast<::mlir::StringAttr>(attr).getValue();
        return value == "i64" || value == "u64" || value == "f64" ||
               value == "i128";
      })
    }]>
  ]>;
}

class Quill_Op<string mnemonic, list<Trait> traits = []>
    : Op<Quill_Dialect, mnemonic, traits>;

def Quill_SourceArrowBatchOp
    : Quill_Op<"source.arrow_batch", [Pure]> {
  let summary = "Arrow RecordBatch source";
  let results = (outs Quill_Batch:$batch);
  let assemblyFormat = "attr-dict `:` type($batch)";
}

def Quill_SourceArrowStreamOp
    : Quill_Op<"source.arrow_stream", [Pure]> {
  let summary = "Arrow RecordBatch stream source";
  let results = (outs Quill_Batch:$batch);
  let assemblyFormat = "attr-dict `:` type($batch)";
}

def Quill_FilterOp
    : Quill_Op<"exec.filter", [SingleBlock]> {
  let summary = "evaluate a row predicate and produce a selection";
  let arguments = (ins Quill_Batch:$batch);
  let regions = (region SizedRegion<1>:$predicate);
  let results = (outs Quill_Selection:$selection);
  let hasVerifier = 1;
  let assemblyFormat = "$batch $predicate attr-dict `:` type($batch) `->` type($selection)";
}

def Quill_ProjectOp
    : Quill_Op<"exec.project", [SingleBlock]> {
  let summary = "project selected rows into a new batch";
  let arguments = (ins Quill_Batch:$batch, Quill_Selection:$selection);
  let regions = (region SizedRegion<1>:$projector);
  let results = (outs Quill_Batch:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $batch `,` $selection $projector attr-dict `:`
    type($batch) `,` type($selection) `->` type($result)
  }];
}

def Quill_LimitOp
    : Quill_Op<"exec.limit", [Pure]> {
  let summary = "truncate a selection";
  let arguments = (ins Quill_Selection:$selection, I64Attr:$fetch);
  let results = (outs Quill_Selection:$result);
  let assemblyFormat = "$selection attr-dict `:` type($selection) `->` type($result)";
}

def Quill_GroupIdsOp
    : Quill_Op<"exec.group_ids", [SingleBlock]> {
  let summary = "bind selected rows to dense runtime group ids";
  let arguments = (ins Quill_Batch:$batch, Quill_Selection:$selection);
  let regions = (region SizedRegion<1>:$keys);
  let results = (outs Quill_GroupIds:$group_ids);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $batch `,` $selection $keys attr-dict `:`
    type($batch) `,` type($selection) `->` type($group_ids)
  }];
}

def Quill_RecordBatchSinkOp
    : Quill_Op<"sink.record_batch", []> {
  let summary = "materialize a batch result";
  let arguments = (ins Quill_Batch:$batch, Quill_Selection:$selection);
  let hasVerifier = 1;
  let assemblyFormat = "$batch `,` $selection attr-dict `:` type($batch) `,` type($selection)";
}

def Quill_PlainSumSinkOp
    : Quill_Op<"sink.plain_sum", [SingleBlock]> {
  let summary = "consume selected rows into a scalar SUM state";
  let arguments = (ins Quill_Batch:$batch, Quill_Selection:$selection);
  let regions = (region SizedRegion<1>:$measure);
  let results = (outs Quill_Scalar:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $batch `,` $selection $measure attr-dict `:`
    type($batch) `,` type($selection) `->` type($result)
  }];
}

def Quill_GroupUpdateSinkOp
    : Quill_Op<"sink.group_update", [SingleBlock]> {
  let summary = "consume selected rows into grouped aggregate state";
  let arguments = (ins Quill_Batch:$batch, Quill_Selection:$selection,
                   Quill_GroupIds:$group_ids,
                   Quill_AggregateFuncArrayAttr:$aggregate_funcs,
                   Quill_AggregateStateTypeArrayAttr:$state_types);
  let regions = (region SizedRegion<1>:$state);
  let results = (outs Quill_Batch:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $batch `,` $selection `,` $group_ids $state attr-dict `:`
    type($batch) `,` type($selection) `,` type($group_ids) `->` type($result)
  }];
}

def Quill_ColumnOp
    : Quill_Op<"column", [Pure]> {
  let summary = "read a column from the current row";
  let arguments = (ins Quill_Row:$row, I64Attr:$index);
  let results = (outs AnyType:$value);
  let hasVerifier = 1;
  let assemblyFormat = "$row attr-dict `:` type($row) `->` type($value)";
}

def Quill_YieldOp
    : Quill_Op<"yield", [Terminator]> {
  let summary = "return values from a Quill pipeline region";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

#endif // QUILL_IR_QUILLOPS_TD
