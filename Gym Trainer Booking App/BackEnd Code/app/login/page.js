"use client";
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, CheckCircle2 } from "lucide-react";

const LoginScreen = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      setError('Both fields are required');
      return;
    }

    setError('');
    const user = { email, password };

    try {
      const response = await fetch('/api/users/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(user),
      });

      const result = await response.json();

      if (response.ok && result.user) {
        localStorage.setItem('currentUser', JSON.stringify(result.user));
        setSuccess('Login Successful');

        if (result.user.isAdmin) {
          router.push('/admin');
        } else if (result.user.isTrainer) {
          router.push('/trainer');
        } else {
          router.push('/home');
        }
      } else {
        setError(result.message || 'Invalid email or password');
      }
    } catch (error) {
      console.error('Error logging in:', error);
      setError('Error logging in. Please try again.');
    }
  };

  return (
    <div className="container mx-auto p-4 flex justify-center items-center min-h-screen">
      <Card className="w-[350px]">
        <CardHeader>
          <CardTitle>Login</CardTitle>
          <CardDescription>Enter your credentials to access your account.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>
                  {typeof error === 'string' ? error : 'An unknown error occurred'}
                </AlertDescription>
              </Alert>
            )}
            {success && (
              <Alert variant="success">
                <CheckCircle2 className="h-4 w-4" />
                <AlertTitle>Success</AlertTitle>
                <AlertDescription>{success}</AlertDescription>
              </Alert>
            )}
            <Button type="submit" className="w-full">
              Login
            </Button>
          </form>
        </CardContent>
        <CardFooter>
          <p className="text-sm text-center w-full">
            No Account? <Link href="/register" className="text-blue-500 hover:underline">Register Here</Link>
          </p>
        </CardFooter>
      </Card>
    </div>
  );
};

export default LoginScreen;