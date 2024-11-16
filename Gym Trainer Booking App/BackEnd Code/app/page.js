import * as React from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function MainPage() {
  return (
    <main className="container mx-auto p-4 flex justify-center items-center min-h-screen">
      <Card className="w-[350px]">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">Personal Trainer App</CardTitle>
          <CardDescription className="text-center">Welcome to your fitness journey</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button asChild className="w-full">
            <Link href="/login">Login</Link>
          </Button>
          <Button asChild variant="outline" className="w-full">
            <Link href="/register">Register</Link>
          </Button>
        </CardContent>
        <CardFooter className="text-center text-sm text-gray-500">
          Start your fitness journey today!
        </CardFooter>
      </Card>
    </main>
  );
}