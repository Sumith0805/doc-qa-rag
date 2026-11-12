// The /login page. In the App Router, a folder name is the URL and page.tsx is what it shows.
import AuthForm from "@/components/AuthForm";

export default function LoginPage() {
  return <AuthForm mode="login" />;
}
