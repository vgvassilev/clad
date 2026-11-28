var Module = {
    preRun: [
        function () {
            const fs = require('fs');
            const path = require('path');

            const directories = [
                {
                    source: '@CLAD_SOURCE_DIR@',
                    destination: '/lib/Differentiator'
                },
                {
                    source: '@LLVM_BUILD_MAIN_INCLUDE_DIR@',
                    destination: '/include/llvm'
                },
                {
                    source: '@CLANG_INCLUDE_DIR@',
                    destination: '/include/clang'
                }
            ];

            function copyDirectory(src, dest) {
                FS.mkdirTree(dest);

                for (const entry of fs.readdirSync(src, {
                    withFileTypes: true
                })) {
                    const srcPath = path.join(src, entry.name);
                    const destPath = dest + '/' + entry.name;

                    if (entry.isDirectory()) {
                        copyDirectory(srcPath, destPath);
                    } else {
                        FS.writeFile(
                            destPath,
                            fs.readFileSync(srcPath)
                        );
                    }
                }
            }

            for (const directory of directories) {
                copyDirectory(
                    directory.source,
                    directory.destination
                );
            }
        }
    ]
};
