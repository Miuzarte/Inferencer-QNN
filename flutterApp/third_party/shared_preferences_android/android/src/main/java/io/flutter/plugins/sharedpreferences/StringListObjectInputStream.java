// Copyright 2013 The Flutter Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// NOTE: This file was restored from the pre-2.4.x Java implementation.
// In 2.4.x upstream migrated this class to Kotlin, but the remaining Java
// file (LegacySharedPreferencesPlugin.java) still references it, and javac
// does not see the Kotlin class in this build configuration
// (flutter/flutter issue #165850). The Kotlin copy was removed to avoid a
// duplicate class.

package io.flutter.plugins.sharedpreferences;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * An ObjectInputStream that only allows string lists, to prevent injected prefs from instantiating
 * arbitrary objects.
 */
class StringListObjectInputStream extends ObjectInputStream {
  private static final Set<String> ALLOWED_CLASS_NAMES =
      new HashSet<>(
          Arrays.asList(
              "java.util.Arrays$ArrayList",
              "java.util.ArrayList",
              "java.lang.String",
              "[Ljava.lang.String;"));

  StringListObjectInputStream(InputStream in) throws IOException {
    super(in);
  }

  @Override
  protected Class<?> resolveClass(ObjectStreamClass desc)
      throws IOException, ClassNotFoundException {
    if (!ALLOWED_CLASS_NAMES.contains(desc.getName())) {
      throw new ClassNotFoundException(desc.getName());
    }
    return super.resolveClass(desc);
  }
}
