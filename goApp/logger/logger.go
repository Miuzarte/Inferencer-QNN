package logger

import (
	"encoding/json"
	"io"
	"os"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

var (
	lastLogOutMonth int
	lastLogOutDay   int
	lastLogOutMu    sync.Mutex
)

func init() {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

// SetGlobalLevel 覆盖默认的 Debug 级别（例如 -v 时提升到 Trace）。
func SetGlobalLevel(level zerolog.Level) {
	zerolog.SetGlobalLevel(level)
}

func New(scope string) zerolog.Logger {
	return NewWithOutput(scope, os.Stdout)
}

func NewWithOutput(scope string, out io.Writer) zerolog.Logger {
	cw := zerolog.ConsoleWriter{
		Out:             out,
		FormatTimestamp: formatTimestamp,
		FormatLevel:     formatLevel,
	}
	return zerolog.New(cw).
		With().
		Timestamp().
		Str("scope", scope).
		Logger()
}

func formatTimestamp(i any) (s string) {
	var t time.Time

	switch tt := i.(type) {
	case string:
		ts, err := time.ParseInLocation(time.RFC3339, tt, time.Local)
		if err != nil {
			return tt
		}
		t = ts
	case json.Number:
		timestamp, err := tt.Int64()
		if err != nil {
			return tt.String()
		}
		t = time.Unix(timestamp, 0)
	default:
		return "<nil>"
	}

	month, day := int(t.Month()), t.Day()
	lastLogOutMu.Lock()
	defer lastLogOutMu.Unlock()
	if month != lastLogOutMonth || day != lastLogOutDay {
		s = t.Format("[01/02 15:04:05]")
	} else {
		s = t.Format("[15:04:05]")
	}
	lastLogOutMonth, lastLogOutDay = month, day
	return
}

func formatLevel(i any) string {
	var level string
	if ll, ok := i.(string); ok {
		level = ll
	} else {
		return ""
	}

	switch level {
	case zerolog.LevelTraceValue:
		return "\x1b[94m[TRACE]\x1b[m"
	case zerolog.LevelDebugValue:
		return "\x1b[92m[DEBUG]\x1b[m"
	case zerolog.LevelInfoValue:
		return "\x1b[97m [INFO]\x1b[m"
	case zerolog.LevelWarnValue:
		return "\x1b[93m [WARN]\x1b[m"
	case zerolog.LevelErrorValue:
		return "\x1b[91m[ERROR]\x1b[m"
	case zerolog.LevelFatalValue:
		return "\x1b[91;5m[FATAL]\x1b[m"
	case zerolog.LevelPanicValue:
		return "\x1b[91;5;7m[PANIC]\x1b[m"
	default:
		return "[" + level + "]"
	}
}
