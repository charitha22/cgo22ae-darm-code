; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -jump-threading -S | FileCheck %s

; CHECK-ALL-LABEL: @func(

define i1 @func(i1 %arg, i32 %arg1, i1 %arg2) {
; CHECK-LABEL: @func(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    br i1 [[ARG:%.*]], label [[BB3:%.*]], label [[BB4:%.*]]
; CHECK:       bb3:
; CHECK-NEXT:    [[I:%.*]] = icmp eq i32 [[ARG1:%.*]], 0
; CHECK-NEXT:    br label [[BB7:%.*]]
; CHECK:       bb4:
; CHECK-NEXT:    callbr void asm sideeffect "", "X"(i8* blockaddress(@func, [[BB7]]))
; CHECK-NEXT:    to label [[BB5:%.*]] [label %bb7]
; CHECK:       bb5:
; CHECK-NEXT:    br label [[BB7]]
; CHECK:       bb7:
; CHECK-NEXT:    [[I8:%.*]] = phi i1 [ [[I]], [[BB3]] ], [ [[ARG2:%.*]], [[BB5]] ], [ [[ARG2]], [[BB4]] ]
; CHECK-NEXT:    [[I9:%.*]] = xor i1 [[I8]], [[ARG]]
; CHECK-NEXT:    br i1 [[I9]], label [[BB11:%.*]], label [[BB11]]
; CHECK:       bb11:
; CHECK-NEXT:    ret i1 [[I9]]
;
bb:
  br i1 %arg, label %bb3, label %bb4

bb3:
  %i = icmp eq i32 %arg1, 0
  br label %bb7

bb4:
  callbr void asm sideeffect "", "X"(i8* blockaddress(@func, %bb6))
  to label %bb5 [label %bb6]

bb5:
  br label %bb6

bb6:
  br label %bb7

bb7:
  %i8 = phi i1 [ %i, %bb3 ], [ %arg2, %bb6 ]
  %i9 = xor i1 %i8, %arg
  br i1 %i9, label %bb11, label %bb10

bb10:
  br label %bb11

bb11:
  ret i1 %i9
}
