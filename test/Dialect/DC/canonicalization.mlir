// RUN: circt-opt %s --canonicalize | FileCheck %s

hw.module @sync(%a: !dc.token, %b : !dc.token) -> (out: !dc.token) {
    %0 = dc.sync %a, %b
    %1 = dc.sync %0
    hw.output %1 : !dc.token
}
