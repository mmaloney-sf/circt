// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{body contains non-pure operation}}
arc.define @Foo(%arg0: i1) {
  // expected-note @+1 {{first non-pure operation here:}}
  arc.state @Bar() clock %arg0 lat 1 : () -> ()
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo() {
  // expected-error @+1 {{'arc.state' op with non-zero latency requires a clock}}
  arc.state @Bar() lat 1 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%clock: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have a clock}}
  arc.state @Bar() clock %clock lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%enable: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have an enable}}
  arc.state @Bar() enable %enable lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @+1 {{body contains non-pure operation}}
arc.define @SupportRecurisveMemoryEffects(%arg0: i1, %arg1: i1) {
  // expected-note @+1 {{first non-pure operation here:}}
  scf.if %arg0 {
    arc.state @Bar() clock %arg1 lat 1 : () -> ()
  }
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @below {{op must have exactly one argument}}
arc.model "MissingArg" {
^bb0:
}

// -----

// expected-error @below {{op must have exactly one argument}}
arc.model "TooManyArgs" {
^bb0(%arg0: !arc.storage, %arg1: !arc.storage):
}

// -----

// expected-error @below {{op argument must be of storage type}}
arc.model "WrongArgType" {
^bb0(%arg0: i32):
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{`Bar` does not reference a valid `arc.define`}}
  arc.call @Bar() : () -> ()
  arc.output
}
func.func @Bar() {
  return
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{incorrect number of operands for arc}}
  arc.call @Bar() : () -> ()
  arc.output
}
arc.define @Bar(%arg0: i1) {
  arc.output
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{incorrect number of results for arc}}
  arc.call @Bar() : () -> ()
  arc.output
}
arc.define @Bar() -> i1 {
  %false = hw.constant false
  arc.output %false : i1
}

// -----

arc.define @Foo(%arg0: i1, %arg1: i32) {
  // expected-error @+3 {{operand type mismatch: operand 1}}
  // expected-note @+2 {{expected type: 'i42'}}
  // expected-note @+1 {{actual type: 'i32'}}
  arc.call @Bar(%arg0, %arg1) : (i1, i32) -> ()
  arc.output
}
arc.define @Bar(%arg0: i1, %arg1: i42) {
  arc.output
}

// -----

arc.define @Foo(%arg0: i1, %arg1: i32) {
  // expected-error @+3 {{result type mismatch: result 1}}
  // expected-note @+2 {{expected type: 'i42'}}
  // expected-note @+1 {{actual type: 'i32'}}
  %0, %1 = arc.call @Bar() : () -> (i1, i32)
  arc.output
}
arc.define @Bar() -> (i1, i42) {
  %false = hw.constant false
  %c0_i42 = hw.constant 0 : i42
  arc.output %false, %c0_i42 : i1, i42
}
