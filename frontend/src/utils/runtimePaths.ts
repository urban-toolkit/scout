const baseUrl = import.meta.env.BASE_URL ?? "/";

const normalizedBase =
  baseUrl === "/" ? "" : baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;

export function appUrl(path: string): string {
  return `${normalizedBase}${path.startsWith("/") ? path : `/${path}`}`;
}
