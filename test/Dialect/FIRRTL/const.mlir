// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "ConstTypes" {
firrtl.module @ConstTypes() {}

// CHECK-LABEL: firrtl.module @ConstUInt(in %a: !firrtl.const.uint<2>) {
firrtl.module @ConstUInt(in %a: !firrtl.const.uint<2>) {}

// CHECK-LABEL: firrtl.module @ConstSInt(in %a: !firrtl.const.sint<2>) {
firrtl.module @ConstSInt(in %a: !firrtl.const.sint<2>) {}

// CHECK-LABEL: firrtl.module @ConstAnalog(in %a: !firrtl.const.analog<2>) {
firrtl.module @ConstAnalog(in %a: !firrtl.const.analog<2>) {}

// CHECK-LABEL: firrtl.module @ConstClock(in %a: !firrtl.const.clock) {
firrtl.module @ConstClock(in %a: !firrtl.const.clock) {}

// CHECK-LABEL: firrtl.module @ConstReset(in %a: !firrtl.const.reset) {
firrtl.module @ConstReset(in %a: !firrtl.const.reset) {}

// CHECK-LABEL: firrtl.module @ConstAsyncReset(in %a: !firrtl.const.asyncreset) {
firrtl.module @ConstAsyncReset(in %a: !firrtl.const.asyncreset) {}

// CHECK-LABEL: firrtl.module @ConstVec(in %a: !firrtl.const.vector<uint<1>, 3>) {
firrtl.module @ConstVec(in %a: !firrtl.const.vector<uint<1>, 3>) {}

// CHECK-LABEL: firrtl.module @ConstBundle(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {
firrtl.module @ConstBundle(in %a: !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {}

// CHECK-LABEL: firrtl.module @MixedConstBundle(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {
firrtl.module @MixedConstBundle(in %a: !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {}

// Subaccess of a const vector should be const only if the index is const
// CHECK-LABEL: firrtl.module @ConstSubaccess
firrtl.module @ConstSubaccess(in %a: !firrtl.const.vector<uint<1>, 3>, in %constIndex: !firrtl.const.uint<4>, in %dynamicIndex: !firrtl.uint<4>) {
  // CHECK-NEXT: [[VAL0:%.+]] = firrtl.subaccess %a[%constIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.const.uint<4>
  // CHECK-NEXT: [[VAL1:%.+]] = firrtl.subaccess %a[%dynamicIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.uint<4>
  // CHECK-NEXT: [[_:%.+]] = firrtl.and [[VAL0]], [[VAL1]] : (!firrtl.const.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %b = firrtl.subaccess %a[%constIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.const.uint<4>
  %c = firrtl.subaccess %a[%dynamicIndex] : !firrtl.const.vector<uint<1>, 3>, !firrtl.uint<4>
  %d = firrtl.and %b, %c : (!firrtl.const.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
}

// Test parsing/printing of multibit mux when constness of operands is mixed
// CHECK-LABEL: firrtl.module @MixedConstMultibitMux
firrtl.module @MixedConstMultibitMux(in %index: !firrtl.uint<2>, in %source_0: !firrtl.const.uint<1>, in %source_1: !firrtl.uint<1>, in %source_2: !firrtl.const.uint<1>) {
  // CHECK-NEXT: [[VAL:%.+]] = firrtl.multibit_mux %index, %source_2, %source_1, %source_0 : !firrtl.uint<2>, !firrtl.const.uint<1>, !firrtl.uint<1>, !firrtl.const.uint<1>
  %0 = firrtl.multibit_mux %index, %source_2, %source_1, %source_0 : !firrtl.uint<2>, !firrtl.const.uint<1>, !firrtl.uint<1>, !firrtl.const.uint<1>
}

}
