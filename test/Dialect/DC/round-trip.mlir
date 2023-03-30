// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module @roundtrip(%a: !dc.token, %b : !dc.token) -> (out: !dc.token) {
    %0 = dc.join %a, %b
    %1, %2  = dc.fork %0 : !dc.token, !dc.token

    dc.symbol @foo
    dc.symbol @bar

    %3 = dc.control %1 [@foo, @bar] : !dc.token
    %4 = dc.data %2 [@foo, @bar] : !dc.token

    hw.output %0 : !dc.token
}
