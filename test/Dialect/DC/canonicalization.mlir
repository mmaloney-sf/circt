// RUN: circt-opt %s --canonicalize | FileCheck %s

hw.module @join(%a: !dc.token, %b : !dc.token) -> (out: !dc.token) {
    %0 = dc.join %a, %b
    %1 = dc.join %0
    hw.output %1 : !dc.token
}

hw.module @fork(%a: !dc.token) -> (out: !dc.token) {
    %0 = dc.fork %a : !dc.token
    hw.output %0 : !dc.token
}
