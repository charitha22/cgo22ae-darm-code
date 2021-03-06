; RUN: llc -mcpu=gfx900 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - %s | llvm-dwarfdump -debug-frame - | FileCheck %s

; CHECK: .debug_frame contents:
; CHECK: 00000000 0000001c ffffffff CIE
; CHECK-NEXT:   Format:                DWARF32
; CHECK-NEXT:   Version:               4
; CHECK-NEXT:   Augmentation:          "[llvm:v0.0]"
; CHECK-NEXT:   Address size:          8
; CHECK-NEXT:   Segment desc size:     0
; CHECK-NEXT:   Code alignment factor: 4
; CHECK-NEXT:   Data alignment factor: 4
; CHECK-NEXT:   Return address column: 16
; CHECK-EMPTY:
; CHECK-NEXT:   DW_CFA_nop:
; CHECK-NEXT:   DW_CFA_nop:
; CHECK-NEXT:   DW_CFA_nop:
; CHECK-NEXT:   DW_CFA_nop:
; CHECK-NEXT:   DW_CFA_nop:
; CHECK-NEXT:   DW_CFA_nop:
; CHECK-EMPTY:
; CHECK-NEXT: 00000020 {{[0-9]+}} 00000000 FDE cie=00000000 pc=00000000...{{[0-9]+}}
; CHECK: .eh_frame contents:

define void @func() #0 {
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "file", directory: "dir")
