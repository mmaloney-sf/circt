// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.const.uint<1>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.bundle<a: const.uint<1>>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.vector<const.uint<1>, 2>) {}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.uint<1>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.const.uint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.uint<1>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.uint<1>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.uint<1>, !firrtl.const.uint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a: !firrtl.const.uint<4>, in %b: !firrtl.const.uint<4>) {
  // expected-error @+1 {{'firrtl.and' op inferred type(s) '!firrtl.const.uint<4>' are incompatible with return type(s) of operation '!firrtl.uint<4>'}}
  %0 = firrtl.and %a, %b : (!firrtl.const.uint<4>, !firrtl.const.uint<4>) -> !firrtl.uint<4>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a: !firrtl.const.uint<4>, in %b: !firrtl.uint<4>) {
  // expected-error @+1 {{'firrtl.and' op inferred type(s) '!firrtl.uint<4>' are incompatible with return type(s) of operation '!firrtl.const.uint<4>'}}
  %0 = firrtl.and %a, %b : (!firrtl.const.uint<4>, !firrtl.uint<4>) -> !firrtl.const.uint<4>
}
}
