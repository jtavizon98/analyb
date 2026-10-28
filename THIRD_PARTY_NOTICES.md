# Third-party notices

## Catppuccin palette

`analyb/colors.py` includes Catppuccin Latte palette values.
Upstream: https://github.com/catppuccin/palette
License source: https://raw.githubusercontent.com/catppuccin/palette/main/LICENSE

```text
MIT License

Copyright (c) 2021 Catppuccin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Units references

`analyb/units.py` uses standard SI prefix names and powers from the BIPM
*SI Brochure*, 9th edition, section 3, Table 7:
https://www.bipm.org/en/publications/si-brochure
Only the existing yocto-through-yotta subset is supported. The unit macro
spellings are compatible with Joseph Wright's *siunitx* manual:
https://ctan.org/pkg/siunitx
These references describe the standard facts and interface conventions;
no siunitx implementation or manual prose is distributed here.
