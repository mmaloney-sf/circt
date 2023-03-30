// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module @sync(%a: !dc.token, %b : !dc.token) -> (out: !dc.token) {
    %0 = dc.sync %a, %b
    hw.output %0 : !dc.token
}
