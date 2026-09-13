package perf

import (
	"strings"
	"testing"
)

func TestLookupProfiles(t *testing.T) {
	cases := []struct {
		name       string
		powerMode  int
		sleepLat   int
		sleepDis   int
		corner     int
		rpcLatency int
		rpcPolling int
		hasDcvs    bool
	}{
		{"default", 0, 0, 0, 0, Unset, Unset, false},
		{"burst", POWER_MODE_PERFORMANCE, 40, 0, VCORNER_MAX, 100, 9999, true},
		{"burst_nosleep", POWER_MODE_PERFORMANCE, 40, 1, VCORNER_MAX, 100, 9999, true},
		{"sustained_high_performance", POWER_MODE_PERFORMANCE, 100, 0, VCORNER_TURBO, 100, 9999, true},
		{"high_performance", POWER_MODE_PERFORMANCE, 100, 0, VCORNER_TURBO, 100, 9999, true},
		{"balanced", POWER_MODE_PERFORMANCE, 1000, 0, VCORNER_NOM_PLUS, 100, 0, true},
		{"power_saver", POWER_MODE_PERFORMANCE, 1000, 0, VCORNER_SVS, 100, 0, true},
	}
	for _, c := range cases {
		got, err := Lookup(c.name)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", c.name, err)
		}
		if got.Name != c.name {
			t.Errorf("%s: Name = %q", c.name, got.Name)
		}
		if got.PowerMode != c.powerMode {
			t.Errorf("%s: PowerMode = %#x, want %#x", c.name, got.PowerMode, c.powerMode)
		}
		if got.SleepLatency != c.sleepLat {
			t.Errorf("%s: SleepLatency = %d, want %d", c.name, got.SleepLatency, c.sleepLat)
		}
		if got.SleepDisable != c.sleepDis {
			t.Errorf("%s: SleepDisable = %d, want %d", c.name, got.SleepDisable, c.sleepDis)
		}
		if got.BusTarget != c.corner || got.CoreTarget != c.corner {
			t.Errorf("%s: corner = %#x/%#x, want %#x", c.name, got.BusTarget, got.CoreTarget, c.corner)
		}
		// 三个电压角必须一致, 否则 DCVS 会在 min/max 之间漂
		if got.BusMin != got.BusMax || got.CoreMin != got.CoreMax {
			t.Errorf("%s: corners not uniform: bus %#x..%#x core %#x..%#x",
				c.name, got.BusMin, got.BusMax, got.CoreMin, got.CoreMax)
		}
		if got.RpcControlLatency != c.rpcLatency {
			t.Errorf("%s: RpcControlLatency = %d, want %d", c.name, got.RpcControlLatency, c.rpcLatency)
		}
		if got.RpcPollingTime != c.rpcPolling {
			t.Errorf("%s: RpcPollingTime = %d, want %d", c.name, got.RpcPollingTime, c.rpcPolling)
		}
		if got.HasDcvs() != c.hasDcvs {
			t.Errorf("%s: HasDcvs = %v, want %v", c.name, got.HasDcvs(), c.hasDcvs)
		}
	}
}

// 非 default 档都必须真的下发点东西, 否则 -htp-perf 是个空操作。
func TestNonDefaultProfilesHaveConfig(t *testing.T) {
	for _, name := range Names() {
		cfg, err := Lookup(name)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", name, err)
		}
		if IsDefault(name) {
			if cfg.HasAny() {
				t.Errorf("default 档不应下发任何配置: %+v", cfg)
			}
			continue
		}
		if !cfg.HasAny() {
			t.Errorf("%s 档没有任何配置项", name)
		}
		if cfg.DcvsEnable != 0 && cfg.DcvsEnable != 1 {
			t.Errorf("%s: DcvsEnable = %d, 只接受 0/1", name, cfg.DcvsEnable)
		}
		if cfg.SleepLatency <= 0 {
			t.Errorf("%s: SleepLatency = %d, 必须为正", name, cfg.SleepLatency)
		}
	}
}

func TestAliases(t *testing.T) {
	for alias, canonical := range aliases {
		a, err := Lookup(alias)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", alias, err)
		}
		b, err := Lookup(canonical)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", canonical, err)
		}
		a.Name, b.Name = "", ""
		if a != b {
			t.Errorf("别名 %q 与 %q 参数不一致: %+v vs %+v", alias, canonical, a, b)
		}
	}
	// 大小写与空白也要能吃下
	for _, s := range []string{"BURST", " burst ", "Burst"} {
		if _, err := Lookup(s); err != nil {
			t.Errorf("Lookup(%q): %v", s, err)
		}
	}
}

func TestUnknownProfile(t *testing.T) {
	_, err := Lookup("turbo")
	if err == nil {
		t.Fatal("未知档位应当报错")
	}
	if !strings.Contains(err.Error(), "default") {
		t.Errorf("错误信息应列出可用档位: %v", err)
	}
}

// 档位顺序表格必须与 profileOrder 对齐, 免得加了档位忘记登记。
func TestNamesOrder(t *testing.T) {
	names := Names()
	if len(names) != len(profiles) {
		t.Fatalf("profileOrder 有 %d 项, profiles 有 %d 项", len(names), len(profiles))
	}
	if names[0] != "default" {
		t.Errorf("default 必须是第一个: %v", names)
	}
	for _, n := range names {
		if _, ok := profiles[n]; !ok {
			t.Errorf("profileOrder 里的 %q 不在 profiles 表中", n)
		}
	}
}
