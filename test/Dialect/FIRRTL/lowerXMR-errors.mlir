// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file -verify-diagnostics

// Test for same module lowering
// CHECK-LABEL: firrtl.circuit "xmr"
firrtl.circuit "xmr" {
  // expected-error @+1 {{reference dataflow cannot be traced back to the remote read op for module port 'a'}}
  firrtl.module @xmr(in %a: !firrtl.ref<uint<2>>) {
    %x = firrtl.ref.resolve %a : !firrtl.ref<uint<2>>
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.const.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.const.uint<1>
    firrtl.refconnect %_a, %1 : !firrtl.ref<uint<1>>, !firrtl.ref<const.uint<1>>
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.ref<uint<1>>)
    %c_b = firrtl.instance child @Child2(in _a: !firrtl.ref<uint<1>>)
    firrtl.refconnect %c_a, %xmr_a : !firrtl.ref<uint<1>>
    firrtl.refconnect %c_b, %xmr_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child1(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    %c_b = firrtl.instance child @Child2(in _a: !firrtl.ref<uint<1>>)
    firrtl.refconnect %c_b, %_a : !firrtl.ref<uint<1>>
  }
  // expected-error @+1 {{op multiply instantiated module with input RefType port '_a'}}
  firrtl.module @Child2(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.const.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.const.uint<1>
    firrtl.refconnect %_a, %1 : !firrtl.ref<uint<1>>, !firrtl.ref<const.uint<1>>
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.ref<uint<1>>)
    %c_b = firrtl.instance child @Child2(in _a: !firrtl.ref<uint<1>>)
    firrtl.refconnect %c_a, %xmr_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child1(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
  }
  // expected-error @+1 {{reference dataflow cannot be traced back to the remote read op for module port '_a'}}
  firrtl.module @Child2(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
  }
}

// -----
// Check handling of unexpected ref.sub.

firrtl.circuit "RefSubNotFromMemory" {
  firrtl.module @RefSubNotFromMemory(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>) {
    // expected-note @below {{input here}}
    %ref = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    // expected-error @below {{can only lower RefSubOp of Memory}}
    %sub = firrtl.ref.sub %ref[1] : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>
    %res = firrtl.ref.resolve %sub : !firrtl.ref<uint<2>>
  }
}

// -----
// Check handling of unexpected ref.sub, from port.

firrtl.circuit "RefSubNotFromOp" {
  // expected-note @below {{input here}}
  firrtl.module private @Child(in %ref : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>) {
    // expected-error @below {{can only lower RefSubOp of Memory}}
    %sub = firrtl.ref.sub %ref[1] : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>
    %res = firrtl.ref.resolve %sub : !firrtl.ref<uint<2>>
  }
  firrtl.module @RefSubNotFromOp(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>) {
    %ref = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %child_ref = firrtl.instance child @Child(in ref : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>)
    firrtl.refconnect %child_ref, %ref : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>
  }
}
