// extract_ctx 从官方 yolo-flutter-app release 的 onnx 中提取 QNN EPContext
// 二进制（onnx 不随仓库分发，需要先自行下载）。
//
// 用法: go run ./cmd/extract_ctx -model models/yolo26n_v73_qnn.onnx -out models/ctx_v73.bin
package main

import (
	"crypto/md5"
	"flag"
	"fmt"
	"os"
)

// mini protobuf wire parser（只覆盖 ONNX ModelProto 需要的字段）
type field struct {
	num int
	wt  int
	val []byte // bytes 字段内容
	vi  uint64 // varint 值
}

func parseFields(b []byte) ([]field, error) {
	var out []field
	for i := 0; i < len(b); {
		tag, n := readVarint(b[i:])
		if n == 0 {
			return nil, fmt.Errorf("bad varint tag at %d", i)
		}
		i += n
		num, wt := int(tag>>3), int(tag&7)
		f := field{num: num, wt: wt}
		switch wt {
		case 0:
			v, n := readVarint(b[i:])
			if n == 0 {
				return nil, fmt.Errorf("bad varint at %d", i)
			}
			f.vi = v
			i += n
		case 2:
			l, n := readVarint(b[i:])
			if n == 0 || i+n+int(l) > len(b) {
				return nil, fmt.Errorf("bad length at %d", i)
			}
			i += n
			f.val = b[i : i+int(l)]
			i += int(l)
		case 1:
			if i+8 > len(b) {
				return nil, fmt.Errorf("truncated fixed64 at %d", i)
			}
			f.val = b[i : i+8]
			i += 8
		case 5:
			if i+4 > len(b) {
				return nil, fmt.Errorf("truncated fixed32 at %d", i)
			}
			f.val = b[i : i+4]
			i += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d at %d", wt, i)
		}
		out = append(out, f)
	}
	return out, nil
}

func readVarint(b []byte) (uint64, int) {
	var v uint64
	for i := 0; i < len(b) && i < 10; i++ {
		x := b[i]
		v |= uint64(x&0x7f) << (7 * i)
		if x&0x80 == 0 {
			return v, i + 1
		}
	}
	return 0, 0
}

func fieldBytes(fs []field, num int) []byte {
	for _, f := range fs {
		if f.num == num && f.wt == 2 {
			return f.val
		}
	}
	return nil
}

func fieldStrings(fs []field, num int) [][]byte {
	var out [][]byte
	for _, f := range fs {
		if f.num == num && f.wt == 2 {
			out = append(out, f.val)
		}
	}
	return out
}

func main() {
	modelPath := flag.String("model", "models/yolo26n_v73_qnn.onnx", "path to official qnn onnx model")
	outPath := flag.String("out", "models/ctx_v73.bin", "output context binary path")
	flag.Parse()

	model, err := os.ReadFile(*modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read model:", err)
		os.Exit(1)
	}

	// ModelProto.graph = 7
	graph := fieldBytes(parseFieldsOrDie(model), 7)
	// GraphProto.node = 1
	var ctx []byte
	for _, node := range repeatedBytes(parseFieldsOrDie(graph), 1) {
		nf := parseFieldsOrDie(node)
		op := string(firstString(nf, 4))
		if op != "EPContext" {
			continue
		}
		for _, attr := range repeatedBytes(nf, 5) {
			af := parseFieldsOrDie(attr)
			name := string(firstString(af, 1))
			if name != "ep_cache_context" {
				continue
			}
			ctx = fieldBytes(af, 4) // AttributeProto.s = 4（新版 onnx.proto）
		}
	}
	if ctx == nil {
		fmt.Fprintln(os.Stderr, "ep_cache_context attribute not found")
		os.Exit(1)
	}
	if err := os.WriteFile(*outPath, ctx, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Printf("extracted %d bytes -> %s (md5 %x)\n", len(ctx), *outPath, md5.Sum(ctx))
}

func parseFieldsOrDie(b []byte) []field {
	fs, err := parseFields(b)
	if err != nil {
		fmt.Fprintln(os.Stderr, "protobuf parse:", err)
		os.Exit(1)
	}
	return fs
}

func repeatedBytes(fs []field, num int) [][]byte {
	var out [][]byte
	for _, f := range fs {
		if f.num == num && f.wt == 2 {
			out = append(out, f.val)
		}
	}
	return out
}

func firstString(fs []field, num int) []byte {
	s := fieldStrings(fs, num)
	if len(s) == 0 {
		return nil
	}
	return s[0]
}
